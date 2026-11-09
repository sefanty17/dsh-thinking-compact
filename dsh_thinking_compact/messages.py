"""Anthropic Messages 格式的压缩适配层。

DSH 的 llm-deepseek 传输走的是 `POST /v1/messages`（Anthropic Messages 兼容），
不是 OpenAI 的 `/v1/chat/completions`：思考在 assistant 消息的 content 里以
`{"type": "thinking", "thinking": ...}` 块出现，工具调用/结果分别是
`tool_use` / `tool_result` 块。`selector.py` 只认 `reasoning_content`，对这套
结构永远选不中，所以这里单独写一套"选一轮 + 出卡片"的最小实现。

安全边界（任何一条不满足就整段不动）：
1. 只切最后一个真人 user 消息之前的完整助手轮；
2. 一个助手轮的 tool_use 必须能在紧随其后的 user 消息里找到全部对应的
   tool_result，否则这个轮次不完整，不切（切了会造成孤立 tool_result）；
3. 最近 keep_rounds 个带工具的轮次保留原文；
4. 切段内思考 token 之和、切段 token 都要过阈值；
5. 卡片是纯 user 文本消息，替换整段（助手轮 + 对应工具结果），
   工具结果原文完整抄进卡片，不丢内容。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .card import MARK, PREAMBLE, render_summary
from .config import Config


@dataclass(frozen=True)
class MsgSpan:
    start: int          # 闭区间，messages 下标
    end: int
    reasoning_tokens: int
    segment_tokens: int
    thinking_text: str

    @property
    def indices(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def cut_reasoning_tokens(self) -> int:
        return self.reasoning_tokens


def _blocks(message: dict) -> list[dict]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _block_text(block: dict) -> str:
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text") or "")
    if kind == "thinking":
        return str(block.get("thinking") or "")
    if kind == "tool_result":
        return _content_text(block.get("content"))
    if kind == "tool_use":
        return json.dumps(block.get("input") or {}, ensure_ascii=False)
    return ""


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_block_text(b) for b in content if isinstance(b, dict))
    return ""


def message_text(message: dict) -> str:
    return "\n".join(_block_text(b) for b in _blocks(message))


def is_human_user(message: dict) -> bool:
    """带真实文本的 user 消息 = 真人指令；纯 tool_result 的 user 不算。"""
    if message.get("role") != "user":
        return False
    for block in _blocks(message):
        if block.get("type") == "text" and str(block.get("text") or "").strip():
            return True
    return False


def thinking_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        for block in _blocks(message):
            if block.get("type") == "thinking" and str(block.get("thinking") or "").strip():
                parts.append(str(block["thinking"]))
    return "\n\n".join(parts)


def _tokens(text: str) -> int:
    return max(1, len(text) // 3) if text else 0


def _tool_use_ids(message: dict) -> list[str]:
    return [
        str(b.get("id") or "")
        for b in _blocks(message)
        if b.get("type") == "tool_use"
    ]


def _tool_result_ids(message: dict) -> set[str]:
    return {
        str(b.get("tool_use_id") or "")
        for b in _blocks(message)
        if b.get("type") == "tool_result"
    }


def _unit_end(messages: list[dict], start: int) -> int | None:
    """一个助手轮的结束下标；工具轮的 tool_result 必须紧随其后且完整。"""
    ids = set(_tool_use_ids(messages[start]))
    if not ids:
        return start
    nxt = start + 1
    if nxt >= len(messages) or messages[nxt].get("role") != "user":
        return None
    if not ids <= _tool_result_ids(messages[nxt]):
        return None
    return nxt


def select(messages: list[dict], cfg: Config | None = None) -> tuple[MsgSpan | None, str]:
    """在当前这一轮指令内部，切掉较早的完整工具轮，保留最近 keep_rounds 个。

    关键点：最后一个真人 user 消息之后的所有助手轮都属于"正在进行的这一轮"，
    不必等到下一条指令才开始压；长程任务正是在一轮里堆出几十个工具轮。
    最新一轮（含当前用户看到的 tool_result）永远保留。
    """
    cfg = cfg or Config.from_env()
    if not messages:
        return None, "no_messages"

    humans = [i for i, m in enumerate(messages) if is_human_user(m)]
    if not humans:
        return None, "no_human_boundary"
    scan = humans[-1] + 1
    if scan >= len(messages):
        return None, "no_closed_round"

    units: list[tuple[int, int]] = []
    i = scan
    while i < len(messages):
        if messages[i].get("role") != "assistant":
            return None, "unexpected_role"
        end = _unit_end(messages, i)
        if end is None:
            return None, "incomplete_tool_round"
        units.append((i, end))
        i = end + 1
    if not units:
        return None, "empty_segment"

    if cfg.keep_rounds <= 0:
        cut_units = units
    elif len(units) <= cfg.keep_rounds:
        return None, "recent_tool_rounds_kept"
    else:
        cut_units = units[:-cfg.keep_rounds]
    if not cut_units:
        return None, "recent_tool_rounds_kept"

    start, end = cut_units[0][0], cut_units[-1][1]
    segment = messages[start:end + 1]
    thinking = thinking_text(segment)
    reasoning_tokens = _tokens(thinking)
    if reasoning_tokens < cfg.min_reasoning_tokens:
        return None, "reasoning_below_threshold"

    segment_tokens = sum(_tokens(message_text(m)) for m in segment)
    if segment_tokens < cfg.chunk_min_tokens:
        return None, "segment_too_small"

    return MsgSpan(start, end, reasoning_tokens, segment_tokens, thinking), "ok"


def _render_tools(segment: list[dict]) -> list[str]:
    lines: list[str] = []
    for message in segment:
        for block in _blocks(message):
            if block.get("type") == "tool_use":
                args = json.dumps(block.get("input") or {}, ensure_ascii=False)
                lines.append("T%d %s(%s)" % (len(lines) + 1, block.get("name") or "?", args))
    return lines


def _render_results(segment: list[dict]) -> list[str]:
    lines: list[str] = []
    for message in segment:
        for block in _blocks(message):
            if block.get("type") == "tool_result":
                lines.append("[T%d] call_id=%s" % (len(lines) + 1, block.get("tool_use_id") or "-"))
                lines.append(_content_text(block.get("content")))
    return lines


def build_card(
    *,
    span: MsgSpan,
    segment: list[dict],
    summary: str,
    backup_id: str,
    digest: str,
    nbytes: int,
) -> str:
    lines = [PREAMBLE, MARK, ""]
    lines.append("[思考] 我当时那一轮的想法（按原话缩句）")
    lines.append("")
    lines.extend(render_summary(summary))
    lines += ["", "[工具调用]"]
    lines.extend(_render_tools(segment) or ["（无）"])
    lines += ["", "[工具结果原文] 一字不改，完整保留"]
    lines.extend(_render_results(segment) or ["（无）"])
    lines += [
        "",
        "[原文指针] seq %d..%d | sha256=%s | bytes=%d | backup=%s"
        % (span.start, span.end, digest, nbytes, backup_id),
    ]
    return "\n".join(lines)


def card_message(card: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": card}]}


def _backup_view(segment: list[dict]) -> list[dict]:
    """把 Messages 结构摊成 shadow.py 认识的 reasoning_content + content。"""
    out: list[dict] = []
    for message in segment:
        think = "\n\n".join(
            str(b.get("thinking") or "") for b in _blocks(message) if b.get("type") == "thinking"
        )
        body = "\n".join(
            _block_text(b) for b in _blocks(message) if b.get("type") != "thinking"
        )
        row = {"role": message.get("role"), "content": body}
        if think:
            row["reasoning_content"] = think
        out.append(row)
    return out


def compact_payload(payload: dict, *, cfg: Config, llm_call) -> tuple[list[dict], "Stats"]:
    """返回 (新 messages, stats)。不满足条件时原样返回，绝不半途改。"""
    from .apply import Stats
    from .shadow import ShadowError, save_shadow
    from .summarizer import summarize

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return messages, Stats(False, "no_messages")
    before = sum(_tokens(message_text(m)) for m in messages)

    span, reason = select(messages, cfg)
    if span is None:
        return messages, Stats(False, reason, tokens_before=before, tokens_after=before)
    if llm_call is None:
        return messages, Stats(False, "no_llm_call", span.indices, before, before)

    segment = messages[span.start:span.end + 1]
    try:
        backup = save_shadow(_backup_view(segment), span, cfg)
    except ShadowError:
        return messages, Stats(False, "backup_failed", span.indices, before, before)

    summary = summarize(span.thinking_text, llm_call, ratio_max=cfg.target_ratio_max)
    if summary is None:
        return messages, Stats(False, "summary_invalid", span.indices, before, before,
                               backup_id=backup.backup_id)

    card = build_card(
        span=span, segment=segment, summary=summary,
        backup_id=backup.backup_id, digest=backup.digest, nbytes=backup.nbytes,
    )
    out = messages[:span.start] + [card_message(card)] + messages[span.end + 1:]
    after = sum(_tokens(message_text(m)) for m in out)
    card_tokens = _tokens(card)
    stats = Stats(
        compacted=True,
        reason="ok",
        span=span.indices,
        tokens_before=before,
        tokens_after=after,
        reasoning_tokens=span.reasoning_tokens,
        card_tokens=card_tokens,
        backup_id=backup.backup_id,
        ratio=span.reasoning_tokens / max(1, card_tokens),
    )
    from .shadow import log_cut
    log_cut(span, backup, card_tokens, cfg)
    return out, stats
