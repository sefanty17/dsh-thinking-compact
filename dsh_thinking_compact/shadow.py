"""原文备份：先落盘，再动消息。写不进去就抛异常，让上层原样保留。

备份格式（纯文本，逐条消息分节）：
    # thinking shadow / messages 2..6
    [role:reasoning] seq=2
    ...思考原文...
    [role:assistant] seq=2
    ...正文 / tool_calls JSON...
    [role:tool] seq=3
    ...工具结果原文...

写文件走 tmp -> os.replace，进程中途挂掉不会留下半截正式文件。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .selector import Span, as_text, reasoning_of


class ShadowError(RuntimeError):
    """备份没写成。调用方必须据此放弃压缩。"""


@dataclass(frozen=True)
class Backup:
    backup_id: str
    digest: str
    nbytes: int
    path: Path


def render_backup(segment: list[dict], span: Span) -> bytes:
    """按 [role:reasoning] / [role:tool] / [role:assistant] 分节渲染。"""
    parts: list[str] = ["# thinking shadow / messages %d..%d" % (span.start, span.end)]
    for offset, message in enumerate(segment):
        seq = span.start + offset
        role = str(message.get("role") or "?")
        reasoning = reasoning_of(message)
        if reasoning:
            parts.append("\n[role:reasoning] seq=%d\n%s" % (seq, reasoning))

        body = as_text(message.get("content"))
        if role == "assistant":
            chunks = [body] if body else []
            if message.get("tool_calls"):
                chunks.append(json.dumps(message["tool_calls"], ensure_ascii=False))
            if chunks:
                parts.append("\n[role:assistant] seq=%d\n%s" % (seq, "\n".join(chunks)))
        elif body:
            parts.append("\n[role:%s] seq=%d\n%s" % (role, seq, body))
    return "\n".join(parts).encode("utf-8")


def save_shadow(segment: list[dict], span: Span, cfg: Config | None = None) -> Backup:
    """备份切段原文，返回 (id, 摘要, 字节数)。任何 IO 失败都抛 ShadowError。"""
    cfg = cfg or Config.from_env()
    raw = render_backup(segment, span)
    digest = hashlib.sha256(raw).hexdigest()
    backup_id = "sp-" + digest[:12]
    final = cfg.shadow_dir / (backup_id + ".txt")
    tmp = cfg.shadow_dir / ("." + backup_id + ".tmp")
    try:
        cfg.shadow_dir.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        if not final.exists() or final.stat().st_size != len(raw):
            raise ShadowError("shadow 备份写入不完整: %s" % final.name)
    except OSError as exc:
        # 清掉可能残留的临时文件，正式文件不存在，等于没备份
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise ShadowError("shadow 备份失败: %s" % exc) from exc
    return Backup(backup_id=backup_id, digest=digest, nbytes=len(raw), path=final)


def log_cut(span: Span, backup: Backup, tokens_after: int, cfg: Config | None = None) -> bool:
    """逐刀明细写 spine-cuts.jsonl：只记序号、token、备份 id，不记原文。

    日志失败不影响这一刀（原文已经在手），返回 False 让调用方决定是否上报。
    """
    cfg = cfg or Config.from_env()
    record = {
        "seq_start": span.start,
        "seq_end": span.end,
        "tokens_before": span.segment_tokens,
        "tokens_after": tokens_after,
        "reasoning_tokens": span.cut_reasoning_tokens,
        "backup": backup.backup_id,
    }
    try:
        cfg.shadow_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.cuts_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return False
    return True
