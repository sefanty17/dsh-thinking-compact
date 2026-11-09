"""缩句：把整轮思考按原话压短，保留第一人称思考口气，不是写纪要。

这里的取舍和"写摘要"不一样：要的是语义压缩，不是摘要删减。覆盖全部思考内容，
压句法、去重复验算、去无信息填充；不得改第三人称、不得改成 JSON/列表/小标题这类
纪要格式。凡是会影响后续判断的事实、权衡、被否方案、数值边界、工具参数，
一条都不能丢。压太狠等于丢失，因此设了 target_ratio_max 上限。

llm_call 是可注入的 (prompt: str) -> str，测试里直接 mock，不联网。
"""
from __future__ import annotations

from typing import Callable

from .config import Config

LlmCall = Callable[[str], str]

PROMPT = """你在压缩一段模型的思考过程。目标：把它"缩句"，让同一个模型读起来觉得这是它自己之前想的，不是别人记的纪要。

【保留】
1. 第一人称和原句骨架，保留思考时的转折口气（"等等""但是""先看看""那就是说""不对"）
2. 每一个判断、决定、取舍，以及支撑它的依据
3. 具体数值、条件、边界、函数名、参数、文件路径

【删除】
1. 重复验算、原地打转、对同一件事反复确认
2. "让我想想""我来分析一下"这类无信息填充
3. 把同一件事说两遍的句子

【禁止】
1. 改成第三人称或旁白口吻（不许写"用户当时希望""助手决定"）
2. 改成 JSON、列表、小标题这类纪要格式
3. 加进原文没有的判断、结论或理由
4. 把"可能""待验证""好像"改写成"确定""已经确认"

【输出】
直接输出缩句后的思考文本，连续段落，不要任何前后缀、解释、标注，不要用引号包裹。

待压缩的思考：

<<<
{TEXT}
>>>
"""


def build_prompt(reasoning_text: str) -> str:
    return PROMPT.replace("{TEXT}", reasoning_text)


def _strip_fence(text: str) -> str:
    """模型偶尔会套 ```，剥掉再校验；不做任何内容修补。"""
    body = text.strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate(summary) -> bool:
    """纯文本校验：非空、且不是 JSON/纪要格式（首字符不是 { 或 [）。"""
    if not isinstance(summary, str) or not summary.strip():
        return False
    if summary.strip()[0] in "{[":
        return False
    return True


def ratio_of(reasoning_text: str, summary: str) -> float:
    """原文 / 缩句 的字符比。越大说明压得越狠，遗漏风险越高。"""
    if not summary:
        return float("inf")
    return len(reasoning_text) / len(summary)


def summarize(
    reasoning_text: str,
    llm_call: LlmCall | None,
    *,
    ratio_max: float | None = None,
) -> str | None:
    """返回缩句后的思考文本；任何一步不合格返回 None（调用方保留原文）。

    不合格的情形：没接模型、调用抛错、返回空、退回 JSON/纪要格式、
    压缩比超过 ratio_max（默认取 Config().target_ratio_max）、缩句反而比原文长。
    """
    if not reasoning_text.strip() or llm_call is None:
        return None
    if ratio_max is None:
        ratio_max = Config().target_ratio_max

    try:
        raw = llm_call(build_prompt(reasoning_text))
    except Exception:
        # 模型超时 / 网络错误 / 输出截断都算不合格，绝不拿半截结果做有损兜底
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None

    summary = _strip_fence(raw).strip()
    if not validate(summary):
        return None
    ratio = ratio_of(reasoning_text, summary)
    if ratio > ratio_max:
        return None
    if ratio < 1.0:
        # 缩句不该比原文还长，说明模型在扩写或加了说明
        return None
    return summary
