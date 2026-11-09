"""主入口：compact_messages。

顺序是固定的，任何一步不合格就整段留下：
    选段 -> 校验前缀指纹 -> 备份原文 -> 生成摘要 -> 生成卡片 -> 再校验前缀 -> 整段替换

替换只发生在"消息数组"这一层：一条 {"role":"user","content":card,"checkpoint":True}
占住原切段的位置。reasoning_content 字段本身一个字都不改，被切的 assistant 消息
是整条从数组里消失，不是被改写。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .card import build_card
from .config import Config
from .selector import reasoning_of, select, text_tokens, total_tokens
from .shadow import ShadowError, log_cut, save_shadow
from .summarizer import LlmCall, summarize


@dataclass(frozen=True)
class Stats:
    compacted: bool
    reason: str                       # ok / 未压缩的具体原因
    span: tuple[int, int] | None = None
    tokens_before: int = 0
    tokens_after: int = 0
    reasoning_tokens: int = 0
    card_tokens: int = 0
    backup_id: str = ""
    ratio: float = 0.0

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


@dataclass(frozen=True)
class CompactResult:
    messages: list[dict]
    stats: Stats
    card: str | None = None

    @property
    def compacted(self) -> bool:
        return self.stats.compacted


def _fingerprint(messages: list[dict]) -> str:
    """切段前缀的字节级指纹。换行、键序都算进去，前缀动过就一定对不上。"""
    digest = hashlib.sha256()
    for message in messages:
        digest.update(
            json.dumps(message, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        )
        digest.update(b"\x00")
    return digest.hexdigest()


def _joined_reasoning(segment: list[dict]) -> str:
    parts: list[str] = []
    for message in segment:
        text = reasoning_of(message)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def compact_messages(
    messages: list[dict],
    *,
    cfg: Config | None = None,
    llm_call: LlmCall | None = None,
) -> CompactResult:
    """尝试把"刚结束的一轮"换成一张卡片；不满足条件就原样返回。"""
    cfg = cfg or Config.from_env()
    before = total_tokens(messages)

    span, reason = select(messages, cfg)
    if span is None:
        return CompactResult(messages, Stats(False, reason, tokens_before=before, tokens_after=before))
    if llm_call is None:
        # 没接模型就不可能产出合格摘要，直接不压，别写空备份
        return CompactResult(
            messages, Stats(False, "no_llm_call", span.indices, before, before)
        )

    segment = messages[span.start:span.end + 1]
    prefix_fp = _fingerprint(messages[:span.start])

    try:
        backup = save_shadow(segment, span, cfg)
    except ShadowError:
        # 备份失败 -> 保留原文，不允许"没备份也照切"
        return CompactResult(
            messages, Stats(False, "backup_failed", span.indices, before, before)
        )

    summary = summarize(_joined_reasoning(segment), llm_call, ratio_max=cfg.target_ratio_max)
    if summary is None:
        return CompactResult(
            messages,
            Stats(False, "summary_invalid", span.indices, before, before, backup_id=backup.backup_id),
        )

    card = build_card(
        span=span,
        segment=segment,
        summary=summary,
        backup_id=backup.backup_id,
        digest=backup.digest,
        nbytes=backup.nbytes,
    )

    # 替换前最后一道闸：切段之前的内容必须与备份前逐字节一致
    if _fingerprint(messages[:span.start]) != prefix_fp:
        raise RuntimeError("compact: 切段前缀在替换前被改动，拒绝替换")

    card_message = {"role": "user", "content": card, "checkpoint": True}
    out = messages[:span.start] + [card_message] + messages[span.end + 1:]

    tail = messages[span.end + 1:]
    if any(out[span.start + 1 + i] is not msg for i, msg in enumerate(tail)):
        raise RuntimeError("compact: 切段之后的消息对象被替换，拒绝提交")
    if any(out[i] is not msg for i, msg in enumerate(messages[:span.start])):
        raise RuntimeError("compact: 切段之前的消息对象被替换，拒绝提交")

    after = total_tokens(out)
    card_tokens = text_tokens(card)
    log_cut(span, backup, card_tokens, cfg)
    ratio = span.cut_reasoning_tokens / max(1, card_tokens)
    stats = Stats(
        compacted=True,
        reason="ok",
        span=span.indices,
        tokens_before=before,
        tokens_after=after,
        reasoning_tokens=span.cut_reasoning_tokens,
        card_tokens=card_tokens,
        backup_id=backup.backup_id,
        ratio=ratio,
    )
    return CompactResult(out, stats, card)


def stats_dict(stats: Stats) -> dict:
    """给日志/代理用的一层扁平结构。"""
    return {
        "compacted": stats.compacted,
        "reason": stats.reason,
        "seq": list(stats.span) if stats.span else None,
        "tokens_before": stats.tokens_before,
        "tokens_after": stats.tokens_after,
        "saved_tokens": stats.saved_tokens,
        "reasoning_tokens": stats.reasoning_tokens,
        "card_tokens": stats.card_tokens,
        "backup": stats.backup_id,
        "reduction": round(1 - stats.tokens_after / stats.tokens_before, 4)
        if stats.tokens_before
        else 0.0,
    }
