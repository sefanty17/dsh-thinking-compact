"""选段：从 OpenAI 格式的 messages 里挑出"刚结束的一轮"。

规则（与 spine.py 的 select_cut 同源，但对象换成了 API messages 数组）：
1. 只切最后一条 user 消息之前、上一个 user 消息之后的那段 assistant/tool 消息；
2. 不跨真实 user 消息；已存在的卡片（带 checkpoint）本身也是边界，不越过；
3. 允许段内包含 tool_calls 与 role=tool —— 工具事件必须和思考一起被整段替换，
   否则 tool 消息会失去它的 assistant 配对；
4. 该轮所有 assistant 的 reasoning token 之和 >= min_reasoning_tokens；
5. 切段 token >= chunk_min_tokens；
6. 切点右边已经有卡片，就不切（那一刀落在已缓存前缀里，白费一次调用）。

token 一律是估算值：真实项目接宿主计数器，这里按 len(text)//3 近似。
消息若自带 reasoning_tokens 字段，优先用它。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import Config


def as_text(value) -> str:
    """把 content（str / 多段 list / None）压成一个字符串，只用于估 token 和备份。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def reasoning_of(message: dict) -> str:
    """思考原文。两个键都认：OpenAI 兼容层的 reasoning_content 与内部 reasoning。"""
    return as_text(message.get("reasoning_content") or message.get("reasoning") or "")


def text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 3)


def message_tokens(message: dict) -> int:
    n = text_tokens(as_text(message.get("content")))
    n += text_tokens(reasoning_of(message))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        n += text_tokens(json.dumps(tool_calls, ensure_ascii=False))
    return n


def total_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def reasoning_tokens_of(message: dict) -> int:
    """优先用宿主给的精确值，没有才估算。"""
    exact = message.get("reasoning_tokens")
    if isinstance(exact, int) and exact >= 0:
        return exact
    return text_tokens(reasoning_of(message))


@dataclass(frozen=True)
class Span:
    """一个可切的整段：[start, end] 闭区间，下标属于原 messages 数组。"""

    start: int
    end: int
    reasoning_tokens: int       # 整轮 reasoning 之和，阈值判定用
    cut_reasoning_tokens: int   # 实际被切掉的部分的 reasoning 之和
    segment_tokens: int         # 实际被切掉的部分的 token

    @property
    def indices(self) -> tuple[int, int]:
        return self.start, self.end


def is_boundary(message: dict) -> bool:
    """user 消息与卡片都是切段地板；卡片角色虽然是 user，语义上不能越过。"""
    return message.get("role") == "user" or bool(message.get("checkpoint"))


def select(messages: list[dict], cfg: Config | None = None) -> tuple[Span | None, str]:
    """返回 (切段, 原因)。原因只在没切时用，方便日志和 demo。"""
    cfg = cfg or Config.from_env()
    if not messages:
        return None, "no_messages"
    if total_tokens(messages) > cfg.max_context_tokens:
        return None, "context_too_large"

    bounds = [i for i, m in enumerate(messages) if is_boundary(m)]
    if not bounds:
        return None, "no_user_boundary"

    last = bounds[-1]
    # 最后一条是卡片：说明这一轮已经被压过，切点右边已有卡片
    if messages[last].get("checkpoint"):
        return None, "card_on_the_right"
    if any(m.get("checkpoint") for m in messages[last + 1:]):
        return None, "card_on_the_right"
    if len(bounds) < 2:
        return None, "no_closed_round"

    start = bounds[-2] + 1
    end = last - 1
    if start > end:
        # 上一个边界是卡片，说明这一轮已经被压过，没有可切的 assistant/tool 段
        if messages[bounds[-2]].get("checkpoint"):
            return None, "card_on_the_right"
        return None, "empty_segment"

    whole = messages[start:end + 1]
    round_reasoning = sum(
        reasoning_tokens_of(m) for m in whole if m.get("role") == "assistant"
    )
    if round_reasoning < cfg.min_reasoning_tokens:
        return None, "reasoning_below_threshold"

    # 最近的 keep_rounds 个工具轮留原文：把右边界退到倒数第 keep_rounds 个
    # 带 tool_calls 的 assistant 之前。退到的位置永远是 assistant 前一位，
    # 因此不会出现"切走 tool_calls 却留下 tool 结果"的断配。
    tool_rounds = [
        i for i in range(start, end + 1)
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")
    ]
    if tool_rounds and cfg.keep_rounds > 0:
        if len(tool_rounds) <= cfg.keep_rounds:
            return None, "recent_tool_rounds_kept"
        end = tool_rounds[-cfg.keep_rounds] - 1
        if end < start:
            return None, "recent_tool_rounds_kept"

    segment = messages[start:end + 1]
    cut_reasoning = sum(
        reasoning_tokens_of(m) for m in segment if m.get("role") == "assistant"
    )
    if cut_reasoning == 0:
        return None, "no_reasoning_in_cut"
    segment_tokens = total_tokens(segment)
    if segment_tokens < cfg.chunk_min_tokens:
        return None, "segment_too_small"

    return (
        Span(
            start=start,
            end=end,
            reasoning_tokens=round_reasoning,
            cut_reasoning_tokens=cut_reasoning,
            segment_tokens=segment_tokens,
        ),
        "ok",
    )


def select_span(messages: list[dict], cfg: Config | None = None) -> Span | None:
    """只要切段，不要原因。"""
    return select(messages, cfg)[0]
