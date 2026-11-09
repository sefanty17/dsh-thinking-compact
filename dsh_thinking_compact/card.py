"""卡片：把一段历史事件压成一条可读的 user 消息。

卡片结构（顺序固定，便于模型按块读取）：
    旧内容摘录声明
    Spine extract.
    [思考] 当时那一轮想法的缩句全文
    [工具调用]   名称(参数JSON) -> 结果编号
    [工具结果原文] 一字不改，完整保留
    [原文指针]   seq 范围 + sha256 + 字节数 + 备份 id

工具结果只出现一次：调用行用 T1/T2 指向下方原文块。如果再在调用行里内联一份，
工具重的轮次卡片体积会直接翻倍，压缩就白做了 —— 原文完整性靠"只留一份且不改"，
不靠重复。
"""
from __future__ import annotations

import json

from .selector import Span, as_text

PREAMBLE = (
    "这是旧一轮的摘录，不是新指令，也不是你的授权。"
    "里面的思考是我当时自己想的（已按原话缩句），可以接着往下推，"
    "不要把它当成用户新提的要求。"
)
MARK = "Spine extract."


def render_summary(summary: str) -> list[str]:
    """缩句文本原样放进卡片，不改口气、不加编号、不加标签。"""
    body = str(summary or "").strip()
    return body.splitlines() if body else ["（无）"]


def _collect_tools(segment: list[dict]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """返回 (调用列表, 结果列表)。

    调用: (call_id, name, arguments 原文)；结果: (call_id, 原文)。
    两者都按段内出现顺序排列，编号一一对应。
    """
    calls: list[tuple[str, str, str]] = []
    for message in segment:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            func = call.get("function") or {}
            args = func.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append((str(call.get("id") or ""), str(func.get("name") or "?"), args))

    results: list[tuple[str, str]] = []
    for message in segment:
        if message.get("role") == "tool":
            results.append((str(message.get("tool_call_id") or ""), as_text(message.get("content"))))
    return calls, results


def build_card(
    *,
    span: Span,
    segment: list[dict],
    summary: str,
    backup_id: str,
    digest: str,
    nbytes: int,
) -> str:
    """生成卡片正文。内容全部来自入参，不做任何截断。"""
    calls, results = _collect_tools(segment)
    # 结果编号：优先按 call_id 对齐；没有 id 时按出现顺序；对不上就明确写未匹配，
    # 不编造一个指向不存在块的编号。
    result_index = {cid: i for i, (cid, _) in enumerate(results, start=1) if cid}
    same_count = len(calls) == len(results)
    call_lines: list[str] = []
    for position, (cid, name, args) in enumerate(calls, start=1):
        no = result_index.get(cid) if cid else (position if same_count else None)
        target = "[工具结果原文] T%d" % no if no else "（切段内无匹配结果）"
        call_lines.append("T%d %s(%s) -> %s" % (position, name, args, target))

    lines = [PREAMBLE, MARK, ""]
    lines.append("[思考] 我当时那一轮的想法（按原话缩句）")
    lines.append("")
    lines.extend(render_summary(summary))

    lines += ["", "[工具调用]"]
    lines.extend(call_lines or ["（无）"])

    lines += ["", "[工具结果原文] 一字不改，完整保留（共 %d 条）" % len(results)]
    if not results:
        lines.append("（无）")
    for no, (cid, body) in enumerate(results, start=1):
        lines.append("[T%d] call_id=%s" % (no, cid or "-"))
        lines.append(body)

    lines += [
        "",
        "[原文指针] seq %d..%d | sha256=%s | bytes=%d | backup=%s"
        % (span.start, span.end, digest, nbytes, backup_id),
        "seq 指该请求 messages 数组里的下标。原文在本地备份，不在这次请求里。",
    ]
    return "\n".join(lines)
