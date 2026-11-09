"""最小 HTTP 代理：转发 DSH 的模型请求，转发前压一次思考链。

DSH 的 deepseek-official 走 Anthropic Messages（`POST /v1/messages`，SSE 流式），
不是 OpenAI 的 `/v1/chat/completions`；两条路由这里都接：

    POST /v1/messages            -> https://api.deepseek.com/anthropic/v1/messages
    POST /v1/chat/completions    -> https://api.deepseek.com/v1/chat/completions

流程（Messages 路由）：
      -> 解析 body，取 messages
      -> messages.compact_payload（摘要走同一条上游链路，非流式小请求）
      -> 原样转发（响应逐块透传，不缓冲整段，否则客户端会触发 SSE idle 超时）
      -> 边转发边收集 SSE，结束后把 usage 追加到 usage.jsonl

已知边界：
- 压缩失败一律放行原文，不改写请求体里任何已有字段。
- 摘要调用不计入 bench 的 token/费用，但它的耗时单独记录在 elapsed/compact_ms。
- Messages 的 reasoning token 明细上游给了就记（reasoning_tokens），没给记 0；
  thinking 字符数仍由调用方从 DSH 的 `--json` 事件里另行统计。

启动：
    python -m dsh_thinking_compact.proxy --port 8787
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

from .apply import compact_messages, stats_dict
from .config import Config

CHAT_PATH = "/v1/chat/completions"
MESSAGES_PATH = "/v1/messages"

# 逐跳头不能转发；Host 要由目标站点决定，Content-Length 由 urllib 重算
DROP_HEADERS = {
    "host", "content-length", "connection", "transfer-encoding",
    "accept-encoding", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "upgrade",
}

# SSE 记账缓冲区上限，防止异常长流把内存吃光
MAX_SSE_BUFFER = 64 * 1024 * 1024


def _open(url: str, body: bytes, headers: dict[str, str], timeout: float):
    req = request.Request(url, data=body, method="POST")
    for key, value in headers.items():
        if key.lower() in DROP_HEADERS:
            continue
        req.add_header(key, value)
    return request.urlopen(req, timeout=timeout)


def _forward(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes, str]:
    """非流式转发，返回 (状态码, 响应体, content-type)。"""
    try:
        with _open(url, body, headers, timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")


def make_llm_call(upstream: str, headers: dict[str, str], model: str, timeout: float,
                  messages_route: bool = False):
    """把摘要请求也打到同一条上游链路上，复用客户端的鉴权头。

    Messages 路由必须走 Messages 协议；这里按路由拼 URL 与 body。
    """
    def call(prompt: str) -> str:
        if messages_route:
            root = upstream.rstrip("/")
            if not root.endswith("/anthropic"):
                root += "/anthropic"
            url = root + MESSAGES_PATH
            payload = {
                "model": model,
                "max_tokens": 16384,
                "stream": False,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            }
        else:
            url = upstream.rstrip("/") + CHAT_PATH
            payload = {
                "model": model,
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "stream": False,
            }
        status, body, _ = _forward(url, json.dumps(payload).encode("utf-8"), headers, timeout)
        if status >= 300:
            raise RuntimeError("summarizer upstream %d" % status)
        data = json.loads(body)
        # 输出被截断 = 缩句不完整，抛错让 summarize() 放弃这次压缩、整段保留原文
        if data.get("stop_reason") == "max_tokens":
            raise RuntimeError("summarizer output truncated")
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            if choices[0].get("finish_reason") == "length":
                raise RuntimeError("summarizer output truncated")
        if "content" in data and isinstance(data["content"], list):
            return "".join(
                str(b.get("text") or "") for b in data["content"] if isinstance(b, dict)
            )
        return data["choices"][0]["message"].get("content") or ""

    return call


def usage_record(payload: dict, response: dict) -> dict:
    """把 chat/completions 的 usage 摊平成一行 JSON。"""
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "ts": round(time.time(), 3),
        "model": response.get("model") or payload.get("model") or "",
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
        "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens", 0),
    }


def sse_usage(buf: bytes) -> dict | None:
    """从 Anthropic SSE 里取 usage：message_start 给输入，message_delta 给输出。"""
    text = buf.decode("utf-8", "replace")
    input_tokens = cache_read = cache_creation = output_tokens = reasoning_tokens = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            usage = (event.get("message") or {}).get("usage") if isinstance(event.get("message"), dict) else None
        if not isinstance(usage, dict):
            continue
        if event.get("type") == "message_start":
            input_tokens = usage.get("input_tokens", input_tokens)
            cache_read = usage.get("cache_read_input_tokens", cache_read)
            cache_creation = usage.get("cache_creation_input_tokens", cache_creation)
            reasoning_tokens = usage.get("reasoning_tokens", reasoning_tokens)
        if event.get("type") == "message_delta":
            output_tokens = usage.get("output_tokens", output_tokens)
            input_tokens = usage.get("input_tokens", input_tokens)
            cache_read = usage.get("cache_read_input_tokens", cache_read)
            cache_creation = usage.get("cache_creation_input_tokens", cache_creation)
            reasoning_tokens = usage.get("reasoning_tokens", reasoning_tokens)
    if input_tokens is None and output_tokens is None:
        return None
    return {
        "input_tokens": int(input_tokens or 0),
        "cache_read_input_tokens": int(cache_read or 0),
        "cache_creation_input_tokens": int(cache_creation or 0),
        "output_tokens": int(output_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "dsh-thinking-compact/1.1"
    protocol_version = "HTTP/1.1"

    # 由 serve() 注入
    cfg: Config
    upstream: str
    compact: bool
    stream_timeout: float = 900.0

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[proxy] " + (fmt % args) + "\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0] != "/health":
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        self._send(200, b'{"ok":true}', "application/json")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path not in (CHAT_PATH, MESSAGES_PATH):
            self._send(404, b'{"error":"only /v1/messages and /v1/chat/completions"}', "application/json")
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        if not isinstance(payload, dict):
            self._send(400, b'{"error":"body must be an object"}', "application/json")
            return

        headers = {k: v for k, v in self.headers.items()}
        is_messages = path == MESSAGES_PATH
        if is_messages:
            root = self.upstream.rstrip("/")
            if not root.endswith("/anthropic"):
                root += "/anthropic"
            upstream_url = root + MESSAGES_PATH
        else:
            upstream_url = self.upstream.rstrip("/") + CHAT_PATH

        compact_stats: dict = {"compacted": False, "reason": "disabled"}
        compact_ms = 0.0
        messages = payload.get("messages")
        if self.compact and isinstance(messages, list) and messages:
            started = time.perf_counter()
            try:
                llm_call = make_llm_call(
                    self.upstream, headers, str(payload.get("model") or ""),
                    self.cfg.max_upstream_seconds, messages_route=is_messages,
                )
                if is_messages:
                    from .messages import compact_payload
                    new_messages, stats = compact_payload(
                        payload, cfg=self.cfg, llm_call=llm_call
                    )
                    payload["messages"] = new_messages
                else:
                    result = compact_messages(messages, cfg=self.cfg, llm_call=llm_call)
                    if result is not None:
                        payload["messages"] = result.messages
                        stats = result.stats
                    else:
                        from .apply import Stats
                        stats = Stats(False, "aborted")
                compact_stats = stats_dict(stats)
            except RuntimeError as exc:
                # 前缀被并发改动这类硬错误：不压缩、照常转发
                self.log_message("compact aborted: %s", exc)
                compact_stats = {"compacted": False, "reason": "aborted"}
            except Exception as exc:  # noqa: BLE001 - 压缩绝不能让主请求挂掉
                self.log_message("compact error: %s", exc)
                compact_stats = {"compacted": False, "reason": "error"}
            compact_ms = round((time.perf_counter() - started) * 1000, 1)
            if compact_stats.get("compacted"):
                self.log_message("compact %s", json.dumps(compact_stats, ensure_ascii=False))

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        try:
            # 生成侧单独用更长的超时：DeepSeek 在 thinking 阶段可能长时间不吐字，
            # 用摘要调用的 120s 会把正常的长思考请求掐断。
            resp = _open(upstream_url, body, headers, self.stream_timeout)
        except error.HTTPError as exc:
            data = exc.read()
            self._log_record(payload, path, started, compact_ms, compact_stats, exc.code, None)
            self._send(exc.code, data, exc.headers.get("Content-Type", "application/json"))
            return
        except Exception as exc:  # noqa: BLE001
            self._log_record(payload, path, started, compact_ms, compact_stats, 599, {"error": str(exc)})
            self._send(599, json.dumps({"error": str(exc)}).encode(), "application/json")
            return

        with resp:
            ctype = resp.headers.get("Content-Type", "text/event-stream")
            length_hdr = resp.headers.get("Content-Length")
            streaming = "text/event-stream" in ctype or is_messages
            if not streaming or length_hdr is not None and "text/event-stream" not in ctype:
                data = resp.read()
                try:
                    parsed = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                self._log_record(payload, path, started, compact_ms, compact_stats, resp.status, parsed)
                self._send(resp.status, data, ctype)
                return

            # SSE：逐块透传，同时留一份用于解析 usage
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            buf = bytearray()
            truncated = False
            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                    if len(buf) < MAX_SSE_BUFFER:
                        buf.extend(chunk)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.log_message("client disconnected")
            except (TimeoutError, OSError) as exc:
                # 已经发出 200/分块头，无法再改状态码；只能中断流并记账
                truncated = True
                self.log_message("upstream stream error: %s", exc)
            finally:
                usage = sse_usage(bytes(buf))
                self._log_record(payload, path, started, compact_ms, compact_stats,
                                 resp.status, None, usage, truncated=truncated)

    def _log_record(
        self, payload: dict, path: str, started: float, compact_ms: float,
        compact_stats: dict, status: int, response: dict | None, usage: dict | None = None,
        truncated: bool = False,
    ) -> None:
        try:
            if usage is None and isinstance(response, dict) and path == MESSAGES_PATH:
                u = response.get("usage") or {}
                usage = {
                    "input_tokens": int(u.get("input_tokens") or 0),
                    "cache_read_input_tokens": int(u.get("cache_read_input_tokens") or 0),
                    "cache_creation_input_tokens": int(u.get("cache_creation_input_tokens") or 0),
                    "output_tokens": int(u.get("output_tokens") or 0),
                }
            if usage is not None:
                record = {
                    "ts": round(time.time(), 3),
                    "route": path,
                    "model": payload.get("model") or "",
                    "prompt_tokens": usage["input_tokens"] + usage["cache_read_input_tokens"]
                    + usage["cache_creation_input_tokens"],
                    "completion_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                    "prompt_cache_hit_tokens": usage["cache_read_input_tokens"],
                    "prompt_cache_miss_tokens": usage["input_tokens"]
                    + usage["cache_creation_input_tokens"],
                    "input_tokens": usage["input_tokens"],
                    "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
                    "cache_read_input_tokens": usage["cache_read_input_tokens"],
                    "output_tokens": usage["output_tokens"],
                }
            elif isinstance(response, dict):
                record = usage_record(payload, response)
                record["route"] = path
            else:
                record = {"ts": round(time.time(), 3), "route": path, "model": payload.get("model") or "",
                          "usage_missing": True}
            record.update({
                "http_status": status,
                "truncated": truncated,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "compact_ms": compact_ms,
                "compacted": bool(compact_stats.get("compacted")),
                "compact_reason": compact_stats.get("reason"),
                "compact_stats": compact_stats if compact_stats.get("compacted") else None,
            })
            append_jsonl(self.cfg.usage_path, record)
        except (OSError, ValueError, KeyError):
            # 记账失败不能影响主请求
            pass


def serve(host: str = "127.0.0.1", port: int = 8787, upstream: str = "https://api.deepseek.com",
          cfg: Config | None = None, compact: bool = True,
          stream_timeout: float = 900.0) -> ThreadingHTTPServer:
    cfg = cfg or Config.from_env()
    handler = type("BoundHandler", (Handler,), {
        "cfg": cfg, "upstream": upstream, "compact": compact, "stream_timeout": stream_timeout,
    })
    server = ThreadingHTTPServer((host, port), handler)
    bound_host, bound_port = server.server_address[:2]
    print("listening on http://%s:%d -> %s" % (bound_host, bound_port, upstream), flush=True)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="thinking-compact 代理")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--upstream", default="https://api.deepseek.com")
    parser.add_argument("--state-dir", default=None,
                        help="usage.jsonl 与 shadow 的根目录（默认 ~/.dsh/thinking-compact）")
    parser.add_argument("--stream-timeout", type=float, default=900.0,
                        help="生成侧单个 socket 读的最大空闲秒数")
    parser.add_argument("--no-compact", action="store_true", help="只转发不压缩（对照用）")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    if args.state_dir:
        cfg = Config.from_env(shadow_dir=Path(args.state_dir) / "shadow")
    server = serve(args.host, args.port, args.upstream, cfg, compact=not args.no_compact,
                   stream_timeout=args.stream_timeout)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
