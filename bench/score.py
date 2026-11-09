"""按 bench/steps.yaml 的 acceptance 逐条验收，算每轮通过率与加权总分。

用法：
    python -m bench.score --group a --round 2          # 对当前 workspace 跑第 2 轮
    python -m bench.score --group a --all              # 汇总 results/a.jsonl 里的历史评分
    python -m bench.score --group a --all --weights staged

公式（bench/README.md）：
    total = sum(w_i * r_i) / sum(w_i)
    r_i = 本轮通过条数 / 本轮 acceptance 总条数
默认 w_i = 1；--weights staged 用 1.0 / 1.2 / 1.5 分段。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = ROOT / "bench" / "steps.yaml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _shell_command(cmd: str) -> list[str] | str:
    """acceptance 里有 heredoc，Windows 上必须交给 bash 而不是 cmd.exe。"""
    bash = shutil.which("bash")
    if os.name != "nt":
        return cmd
    if bash:
        return [bash, "-c", cmd]
    return cmd


def run_command(cmd: str, cwd: Path, timeout: float = 120.0) -> dict:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            _shell_command(cmd),
            cwd=str(cwd),
            shell=isinstance(_shell_command(cmd), str) and os.name == "nt",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "exit": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "seconds": round(time.perf_counter() - started, 2),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": "TIMEOUT after %ss" % timeout,
            "seconds": round(time.perf_counter() - started, 2),
        }


def score_round(step, cwd: Path, timeout: float = 120.0) -> dict:
    results = []
    for cmd in step.acceptance:
        res = run_command(cmd, cwd, timeout)
        results.append({
            "command": cmd,
            "ok": res["ok"],
            "exit": res["exit"],
            "seconds": res["seconds"],
            "stderr_head": "\n".join((res["stderr"] or "").splitlines()[:20]),
        })
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    return {
        "round": step.round,
        "title": step.title,
        "passed": passed,
        "total": total,
        "rate": round(passed / total, 4) if total else 0.0,
        "seconds": round(sum(r["seconds"] for r in results), 2),
        "checks": results,
    }


def weight(round_no: int, mode: str = "flat") -> float:
    if mode == "staged":
        if round_no <= 10:
            return 1.0
        if round_no <= 19:
            return 1.2
        return 1.5
    return 1.0


def weighted_total(scores: list[dict], mode: str = "flat") -> dict:
    if not scores:
        return {"total": 0.0, "weight_sum": 0.0, "rounds": 0, "first_failure": None}
    num = sum(weight(s["round"], mode) * s["rate"] for s in scores)
    den = sum(weight(s["round"], mode) for s in scores)
    first = next((s["round"] for s in scores if s["rate"] < 1.0), None)
    return {
        "total": round(num / den, 4) if den else 0.0,
        "weight_sum": round(den, 2),
        "rounds": len(scores),
        "first_failure": first,
        "weights": mode,
    }


def load_recorded(group: str) -> list[dict]:
    path = ROOT / ".bench" / "results" / ("%s.jsonl" % group)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row.get("score"), dict):
            out.append(row["score"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bench 验收打分")
    parser.add_argument("--group", choices=["a", "b"], required=True)
    parser.add_argument("--round", type=int, default=None, help="对当前 workspace 跑指定轮")
    parser.add_argument("--all", action="store_true", help="汇总 results/<group>.jsonl 的历史评分")
    parser.add_argument("--weights", choices=["flat", "staged"], default="flat")
    parser.add_argument("--timeout", type=float, default=120.0, help="单条 acceptance 超时（秒）")
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args(argv)

    from bench.steps import load_steps

    steps = {s.round: s for s in load_steps(STEPS)}
    cwd = Path(args.workspace) if args.workspace else ROOT / ".bench" / ("work-%s" % args.group)

    if args.round is not None:
        if args.round not in steps:
            print("no such round: %s" % args.round, file=sys.stderr)
            return 2
        result = score_round(steps[args.round], cwd, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["rate"] == 1.0 else 1

    scores = load_recorded(args.group)
    print(json.dumps(weighted_total(scores, args.weights), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
