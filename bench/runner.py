"""A/B 长程 harness：同一串 30 轮指令，A=原版 DSH，B=DSH 走本地压缩代理。

每组一个独立 workspace、一个独立 session、一份 results/<group>.jsonl。
逐轮调用 `dsh --profile headless --json`，第一轮抓 session id，之后用
`--session-id` 续同一会话。单轮超时记 failure 后继续，不整体崩。

用法：
    python -m bench.runner --group a --rounds 2
    python -m bench.runner --group b --rounds 2
    python -m bench.runner --group a --rounds 30 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.score import score_round  # noqa: E402
from bench.steps import Step, load_steps, prompt_for  # noqa: E402

STEPS_PATH = ROOT / "bench" / "steps.yaml"
BENCH = ROOT / ".bench"
RESULTS = BENCH / "results"
RAW = BENCH / "raw"
WORKSPACES = {"a": BENCH / "work-a", "b": BENCH / "work-b"}
PATCHES = BENCH / "patches"
PROXY_PORT = 8787
PROXY_LOG = BENCH / "proxy.log"
DEFAULT_TIMEOUT = 1800.0


def proxy_state_dir(group: str) -> Path:
    return BENCH / ("proxy-state-" + group)


def _dsh_command(args: list[str]) -> tuple[list[str], bool]:
    """优先直接调 node + bin.js。

    Windows 上 shutil.which('dsh') 是 dsh.CMD：走 shell 会多一层 cmd.exe，
    timeout 只能杀掉 cmd，node 子进程会继续跑并把管道撑住，runner 就永远卡在
    communicate()。直接起 node 让被杀的进程就是真正干活的进程。
    """
    shim = shutil.which("dsh")
    node = shutil.which("node")
    if shim:
        root = Path(shim).resolve().parent
        bin_js = root / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"
        if node and bin_js.exists():
            return [node, str(bin_js)] + args, False
    return [shim or "dsh"] + args, os.name == "nt"


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, text=True, timeout=20,
        )
    else:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def resolve_api_key() -> str | None:
    """优先环境变量；缺失时读 ~/.dsh/.credentials.yaml 的 refs（只进子进程 env，不落盘）。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    creds = Path.home() / ".dsh" / ".credentials.yaml"
    if not creds.exists():
        return None
    match = re.search(r"^\s*DEEPSEEK_API_KEY:\s*(\S+)\s*$", creds.read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else None


def write_patch() -> Path:
    PATCHES.mkdir(parents=True, exist_ok=True)
    path = PATCHES / "proxy.yml"
    path.write_text(
        "- id: llm-deepseek\n"
        "  config:\n"
        "    baseURL: http://127.0.0.1:%d\n" % PROXY_PORT,
        encoding="utf-8",
    )
    return path


def _healthy() -> bool:
    try:
        with request.urlopen("http://127.0.0.1:%d/health" % PROXY_PORT, timeout=2) as resp:
            return resp.status == 200
    except (error.URLError, OSError):
        return False


def start_proxy(compact: bool, group: str) -> subprocess.Popen | None:
    """返回新进程；若端口已有健康代理则复用并返回 None。"""
    if _healthy():
        print("[runner] proxy already listening on %d, reuse" % PROXY_PORT)
        return None
    d = proxy_state_dir(group)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "dsh_thinking_compact.proxy",
        "--port", str(PROXY_PORT),
        "--state-dir", str(d),
    ]
    if not compact:
        cmd.append("--no-compact")
    env = dict(os.environ)
    key = resolve_api_key()
    if key:
        env["DEEPSEEK_API_KEY"] = key
    log = open(PROXY_LOG, "ab")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=log, env=env)
    deadline = time.time() + 20
    while time.time() < deadline:
        if _healthy():
            print("[runner] proxy started pid=%s log=%s" % (proc.pid, PROXY_LOG))
            return proc
        if proc.poll() is not None:
            raise RuntimeError("proxy exited early (code %s), see %s" % (proc.returncode, PROXY_LOG))
        time.sleep(0.3)
    proc.terminate()
    raise RuntimeError("proxy did not become healthy on port %d" % PROXY_PORT)


def stop_proxy(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def parse_events(stdout: str) -> dict:
    """把 --json 的 NDJSON 事件折成一轮的账。"""
    session_id = None
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
             "cache_write_tokens": 0, "total_tokens": 0,
             "reasoning_tokens": 0, "reasoning_seen": False}
    steps = 0
    thinking_chars = 0
    tool_calls = 0
    final_chars = 0
    error = None
    completed = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "session":
            session_id = event.get("sessionId") or session_id
        elif kind == "status" and event.get("phase") == "step_end":
            steps += 1
            u = event.get("usage") or {}
            usage["input_tokens"] += int(u.get("inputTokens") or 0)
            usage["output_tokens"] += int(u.get("outputTokens") or 0)
            usage["cache_read_tokens"] += int(u.get("cacheReadTokens") or 0)
            usage["cache_write_tokens"] += int(u.get("cacheWriteTokens") or 0)
            usage["total_tokens"] += int(u.get("totalTokens") or 0)
            exact_reasoning = u.get("reasoningTokens") or u.get("reasoning_tokens")
            if exact_reasoning is not None:
                usage["reasoning_tokens"] += int(exact_reasoning or 0)
                usage["reasoning_seen"] = True
        elif kind == "thinking":
            thinking_chars += len(str(event.get("text") or ""))
        elif kind == "tool_call":
            tool_calls += 1
        elif kind == "final":
            final_chars = len(str(event.get("text") or ""))
        elif kind == "status" and event.get("phase") == "turn_end":
            reason = event.get("reason") or {}
            if reason.get("kind") == "completed":
                completed = True
            elif reason.get("kind") == "error":
                err = reason.get("error") or {}
                error = "%s: %s" % (err.get("code") or "ERROR", err.get("message") or "")
    return {
        "session_id": session_id,
        "usage": usage,
        "steps": steps,
        "thinking_chars": thinking_chars,
        "tool_calls": tool_calls,
        "final_chars": final_chars,
        "completed": completed,
        "error": error,
    }


def proxy_snapshot(group: str) -> int:
    path = proxy_state_dir(group) / "usage.jsonl"
    return path.stat().st_size if path.exists() else 0


def proxy_delta(group: str, offset: int) -> dict:
    path = proxy_state_dir(group) / "usage.jsonl"
    if not path.exists():
        return {"calls": 0}
    with open(path, "rb") as fh:
        fh.seek(offset)
        raw = fh.read()
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return {"calls": 0}
    return {
        "calls": len(rows),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
        "cache_hit_tokens": sum(int(r.get("prompt_cache_hit_tokens") or 0) for r in rows),
        "upstream_ms": round(sum(float(r.get("elapsed_ms") or 0) for r in rows), 1),
        "compact_ms": round(sum(float(r.get("compact_ms") or 0) for r in rows), 1),
        "compacted_calls": sum(1 for r in rows if r.get("compacted")),
        "reasoning_tokens": sum(int(r.get("reasoning_tokens") or 0) for r in rows),
        "usage_missing": sum(1 for r in rows if r.get("usage_missing")),
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "compact_reasons": sorted({str(r.get("compact_reason")) for r in rows}),
        "last_compaction": next(
            (r.get("compact_stats") for r in reversed(rows) if r.get("compacted")), None
        ),
    }


def prepare_workspace(group: str, resume: bool) -> Path:
    ws = WORKSPACES[group]
    if not resume:
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / ".bench-workspace").write_text("group=%s\n" % group, encoding="utf-8")
    else:
        if not ws.exists():
            raise SystemExit("--resume 但 workspace 不存在: %s" % ws)
    return ws


def load_state(group: str) -> dict:
    path = RESULTS / ("%s.state.json" % group)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"session_id": None, "completed": []}


def save_state(group: str, state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / ("%s.state.json" % group)).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_result(group: str, record: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / ("%s.jsonl" % group), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _invoke(cmd: list[str], cwd: Path, env: dict, task: str, timeout: float,
            use_shell: bool) -> tuple[int | None, str, str, bool]:
    """用临时文件接 stdin/stdout/stderr，避免管道被孙进程撑住导致永远等不到 EOF。"""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dsh-round-") as td:
        td_path = Path(td)
        task_path, out_path, err_path = td_path / "task.txt", td_path / "out.txt", td_path / "err.txt"
        task_path.write_text(task, encoding="utf-8")
        with open(task_path, "r", encoding="utf-8") as fin, \
                open(out_path, "w", encoding="utf-8") as fout, \
                open(err_path, "w", encoding="utf-8") as ferr:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), env=env, stdin=fin, stdout=fout, stderr=ferr,
                shell=use_shell, text=True, encoding="utf-8", errors="replace",
            )
            try:
                code = proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                code = proc.returncode
                timed_out = True
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        stderr += "\nTIMEOUT after %ss; process tree killed" % timeout
    return code, stdout, stderr, timed_out


def run_round(step: Step, group: str, ws: Path, session_id: str | None,
              timeout: float, patch: Path | None) -> dict:
    task = prompt_for(step)
    args = ["--profile", "headless"]
    if patch is not None:
        args += ["--patch", str(patch)]
    args += ["--json"]
    if session_id:
        args += ["--session-id", session_id]
    args += ["-"]

    cmd, use_shell = _dsh_command(args)
    env = dict(os.environ)
    key = resolve_api_key()
    if key:
        env["DEEPSEEK_API_KEY"] = key

    started = time.perf_counter()
    status = "ok"
    timed_out = False
    stdout = stderr = ""
    exit_code = None
    try:
        exit_code, stdout, stderr, timed_out = _invoke(cmd, ws, env, task, timeout, use_shell)
        if timed_out:
            status = "timeout"
        elif exit_code != 0:
            status = "failed"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        stderr = "runner error: %s" % exc
    wall = round(time.perf_counter() - started, 1)

    parsed = parse_events(stdout)
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / ("%s-r%02d.ndjson" % (group, step.round))
    raw_path.write_text(stdout, encoding="utf-8")
    if not timed_out and not parsed["completed"] and status == "ok":
        status = "failed"
    if parsed["error"]:
        status = "failed"
    return {
        "group": group,
        "round": step.round,
        "title": step.title,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": wall,
        "status": status,
        "exit_code": exit_code,
        "events_path": str(raw_path),
        "session_id": parsed["session_id"] or session_id,
        "steps": parsed["steps"],
        "tool_calls": parsed["tool_calls"],
        "thinking_chars": parsed["thinking_chars"],
        "final_chars": parsed["final_chars"],
        "usage": {
            "input_tokens": parsed["usage"]["input_tokens"],
            "prompt_tokens": parsed["usage"]["input_tokens"] + parsed["usage"]["cache_read_tokens"]
            + parsed["usage"]["cache_write_tokens"],
            "completion_tokens": parsed["usage"]["output_tokens"],
            "reasoning_tokens": (parsed["usage"]["reasoning_tokens"]
                                 if parsed["usage"]["reasoning_seen"]
                                 else parsed["thinking_chars"] // 3),
            "reasoning_tokens_estimated": not parsed["usage"]["reasoning_seen"],
            "cache_hit_tokens": parsed["usage"]["cache_read_tokens"],
            "cache_write_tokens": parsed["usage"]["cache_write_tokens"],
            "total_tokens": parsed["usage"]["total_tokens"],
        },
        "error": parsed["error"] or (stderr.strip().splitlines()[-1] if status == "failed" and stderr.strip() else None),
        "stderr_tail": "\n".join(stderr.strip().splitlines()[-10:]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSH A/B 长程测试 harness")
    parser.add_argument("--group", choices=["a", "b"], required=True)
    parser.add_argument("--rounds", type=int, default=30, help="只跑到第 N 轮（冒烟用 2）")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--no-compact", action="store_true", help="B 组只走代理不压缩（诊断用）")
    parser.add_argument("--no-score", action="store_true", help="本轮不跑 acceptance")
    args = parser.parse_args(argv)

    if not resolve_api_key():
        print("DEEPSEEK_API_KEY 未设置，且 ~/.dsh/.credentials.yaml 里没有 refs.DEEPSEEK_API_KEY",
              file=sys.stderr)
        return 2

    steps = load_steps(STEPS_PATH)
    targets = [s for s in steps if s.round <= args.rounds]
    if not targets:
        print("没有匹配的轮次", file=sys.stderr)
        return 2

    ws = prepare_workspace(args.group, args.resume)
    state = load_state(args.group) if args.resume else {"session_id": None, "completed": []}
    RESULTS.mkdir(parents=True, exist_ok=True)

    patch = write_patch()
    compact = (args.group == "b") and not args.no_compact
    proxy = start_proxy(compact, args.group)

    print("[runner] group=%s rounds=%d workspace=%s session=%s"
          % (args.group, len(targets), ws, state.get("session_id")))
    print("[runner] compact=%s proxy=on" % compact)
    try:
        for step in targets:
            if args.resume and step.round in state["completed"]:
                print("[runner] round %d already recorded, skip" % step.round)
                continue

            offset = proxy_snapshot(args.group)
            print("[runner] round %d/%d start: %s" % (step.round, args.rounds, step.title), flush=True)
            record = run_round(step, args.group, ws, state.get("session_id"),
                               args.timeout, patch)
            record["proxy"] = proxy_delta(args.group, offset)

            if record["session_id"]:
                state["session_id"] = record["session_id"]

            if not args.no_score:
                score = score_round(step, ws)
                record["score"] = score
                record["score_ok"] = score["rate"] == 1.0

            append_result(args.group, record)
            state["completed"] = sorted(set(state["completed"]) | {step.round})
            save_state(args.group, state)

            print("[runner] round %d done: %s %.1fs tokens=%s score=%s"
                  % (step.round, record["status"], record["wall_seconds"],
                     record["usage"]["total_tokens"],
                     record.get("score", {}).get("rate")), flush=True)

            if not record["session_id"]:
                print("[runner] 没拿到 session id，后续轮无法续同一会话，停止", file=sys.stderr)
                return 1
    finally:
        stop_proxy(proxy)

    print("[runner] finished; results -> %s" % (RESULTS / ("%s.jsonl" % args.group)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
