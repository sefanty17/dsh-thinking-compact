"""思维链压缩原型：在响应写入历史之前，把长 reasoning 换成一个固定格式的摘要。

设计要点（详见对话结论）：
1. 压缩产物走"旁路"，不改写 API 协议字段 reasoning_content。
2. 覆盖动作只能发生在 append 之前；一旦消息随请求发出（sent=True），永不再改，
   否则前缀变了，provider 的 prompt cache 全部失效。
3. 带 tools 时 reasoning_content 必须原样回传（DeepSeek 会 400），
   Anthropic 的 signature / Gemini 的 thought_signature 同理不可改。
4. 压缩记忆追加在历史尾部，不要挪到 system 头部，否则每轮都在改前缀。

零依赖，python compact.py 可直接跑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

SENTINEL = "CM_SUMMARY_V1"

# 事实 / 候选 / 决定 是用户要的三段；rejected 和 open 是"保原意"的关键补充：
# 不记被否决的分支和未验证假设，下一轮会重走死路。
SUMMARY_KEYS = ("facts", "options", "decision", "rejected", "open")

SUMMARY_INSTRUCTION = """在回答之前，先输出一个固定格式的思考摘要块，格式严格如下：
<CM_SUMMARY_V1>
{"facts":["已确认的事实..."],"options":["考虑过的候选方案..."],"decision":"最终决定","rejected":["被否决的分支及原因"],"open":["未验证假设/待办"]}
</CM_SUMMARY_V1>
要求：只保留后续轮次做决策所必需的信息；不要复述试错、验算、自我纠正过程；
不要编造原文没有的事实；数组为空时写 []；整个块必须是合法 JSON。
摘要块之后才是给用户看的正文。"""


def tags(nonce: str | None = None) -> tuple[str, str]:
    """哨兵用非自然语言标记，避免和正文里的标题撞车；nonce 可防注入/复述。"""
    suffix = f":{nonce}" if nonce else ""
    return f"<{SENTINEL}{suffix}>", f"</{SENTINEL}{suffix}>"


def split_content(text: str, nonce: str | None = None) -> tuple[str, str | None, str | None]:
    """从正文里剥出摘要块，返回 (干净正文, 摘要原始 JSON, 错误码)。

    降级原则：拿不到完整块就原样返回、跳过压缩，绝不使用半截摘要。
    """
    open_tag, close_tag = tags(nonce)
    start = text.find(open_tag)
    if start < 0:
        return text, None, "missing_open"
    end = text.find(close_tag, start + len(open_tag))
    if end < 0:
        # 通常是输出被 max_tokens 截断：哨兵之后全是半截 JSON，丢掉，别显示给用户。
        # 上层看到 unclosed 应视为本轮响应不完整，压缩和正常使用都跳过。
        return text[:start].rstrip(), None, "unclosed"
    raw = text[start + len(open_tag):end].strip()
    clean = (text[:start] + text[end + len(close_tag):]).lstrip("\n")
    return clean, raw, None


def parse_summary(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not all(k in data for k in SUMMARY_KEYS):
        return None
    if not isinstance(data["decision"], str):
        return None
    for key in ("facts", "options", "rejected", "open"):
        if not isinstance(data[key], list):
            return None
    return data


@dataclass
class Turn:
    role: str
    content: str
    # 协议字段：带 tools 时才保留，且必须原样，永不用摘要覆盖
    reasoning_content: str | None = None
    tool_calls: list | None = None
    # 旁路字段：压缩产物，由 agent 自己决定怎么注入下一轮
    summary: dict | None = None
    compacted: bool = False
    sent: bool = False


Summarizer = Callable[[str], dict | None]


def ingest(
    history: list[Turn],
    *,
    content: str,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    summarizer: Summarizer | None = None,
    with_tools: bool = False,
    nonce: str | None = None,
) -> Turn:
    """唯一的压缩钩子：收到完整响应 -> 立刻切摘要 -> 构造精简 Turn -> append。

    之后这条 Turn 只允许读，不允许改。
    """
    clean, raw, err = split_content(content, nonce)
    summary = parse_summary(raw) if raw else None

    # 模型没自己写摘要块时，走旁路小模型后置压缩（推荐路线）
    if summary is None and reasoning_content and summarizer:
        summary = summarizer(reasoning_content)

    turn = Turn(
        role="assistant",
        content=clean,
        reasoning_content=reasoning_content if with_tools else None,
        tool_calls=tool_calls,
        summary=summary,
        compacted=summary is not None,
    )
    history.append(turn)
    return turn


def to_api_messages(history: list[Turn], *, with_tools: bool = False) -> list[dict]:
    """组装发往 API 的 messages，并标记 sent。

    压缩记忆以独立块追加在该轮之后（尾部追加 = 前缀稳定 = 缓存可命中）。
    """
    out: list[dict] = []
    for turn in history:
        msg: dict = {"role": turn.role, "content": turn.content}
        if with_tools and turn.role == "assistant" and turn.reasoning_content:
            msg["reasoning_content"] = turn.reasoning_content  # 原样，不可压缩
        if turn.tool_calls:
            msg["tool_calls"] = turn.tool_calls
        out.append(msg)
        if turn.summary:
            out.append({
                "role": "user",
                "content": "<cm:memory>\n" + json.dumps(turn.summary, ensure_ascii=False) + "\n</cm:memory>",
            })
        turn.sent = True
    return out


def demo() -> None:
    reasoning = (
        "用户问 9.11 和 9.8 哪个大。先看整数部分都是 9，再看十分位 1 和 8……"
        "等等，会不会是版本号比较？如果是版本号 9.11 比 9.8 新。但用户说的是'哪个大'，"
        "通常是数值比较。我再验算一次：9.8 = 9.80，9.80 > 9.11。确认是数值比较。"
        "另外还要考虑是不是字符串字典序，字典序 '9.8' > '9.11' 也是 9.8 大，结论一致。"
    ) * 6

    answer_with_block = """<CM_SUMMARY_V1>
{"facts":["用户问 9.11 与 9.8 谁更大","语境为数值比较"],"options":["按小数数值比较","按版本号比较","按字符串字典序比较"],"decision":"回答 9.8 更大（9.8 = 9.80 > 9.11）","rejected":["版本号比较：用户用'大'而非'新'，且无版本语境"],"open":[]}
</CM_SUMMARY_V1>
9.8 更大。把小数位对齐就是 9.80 对 9.11，十分位 8 > 1。"""

    history: list[Turn] = [Turn(role="user", content="9.11 和 9.8 哪个大？")]
    turn = ingest(history, reasoning_content=reasoning, content=answer_with_block)

    print("== 切分结果 ==")
    print("给用户的正文 :", turn.content)
    print("是否压缩成功 :", turn.compacted)
    print("摘要 JSON    :", json.dumps(turn.summary, ensure_ascii=False))
    print("原 thinking  :", len(reasoning), "字符")
    print("压缩后摘要   :", len(json.dumps(turn.summary, ensure_ascii=False)), "字符")
    print("压缩率       : %.1f%%" % (100 - 100 * len(json.dumps(turn.summary, ensure_ascii=False)) / len(reasoning)))

    print("\n== 不带 tools：thinking 直接不进历史 ==")
    msgs = to_api_messages(history)
    for m in msgs:
        print(" ", m["role"], "|", str(m["content"])[:60].replace("\n", " "), "| reasoning:", "reasoning_content" in m)

    print("\n== 带 tools：reasoning_content 原样回传，摘要走旁路 ==")
    history2: list[Turn] = [Turn(role="user", content="查一下订单状态")]
    ingest(history2, reasoning_content=reasoning, content="我来查一下。", with_tools=True,
           summarizer=lambda r: {"facts": ["需要调 get_order"], "options": [], "decision": "调用 get_order", "rejected": [], "open": []})
    for m in to_api_messages(history2, with_tools=True):
        kept = m.get("reasoning_content")
        print(" ", m["role"], "| reasoning 原样:", kept is not None and len(kept) == len(reasoning))

    print("\n== 降级：标记未闭合时跳过压缩，丢弃半截块 ==")
    clean, raw, err = split_content("正文前半 <CM_SUMMARY_V1> {\"facts\": [")
    print(" error =", err, "| clean =", clean)


if __name__ == "__main__":
    demo()
