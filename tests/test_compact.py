"""压缩主流程的验收测试。

覆盖：阈值不够不切、含 tool_calls 的轮次能切、备份失败不切、
摘要不合格不切（含退回 JSON 格式）、前缀不变（含并发改动防护）。
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dsh_thinking_compact.apply import compact_messages
from dsh_thinking_compact.config import Config
from dsh_thinking_compact.selector import select, total_tokens
from dsh_thinking_compact.summarizer import validate

UNIT = "核对 A1 的物流节点，承诺 48 小时，实际 72 小时，差值 24 小时；"
TOOL_A = '{"carrier":"SF","tracking_no":"SF1234567890","promise_hours":48}'
TOOL_B = "1. query 节点\n2. 对比 48h\n3. 超过则 urge 催单（备注写运单号）"


def long_reasoning(chars: int = 12000) -> str:
    """拼到指定长度；默认 12000 字符 ≈ 4000 token，刚好压线达到阈值。"""
    times = chars // len(UNIT) + 1
    return (UNIT * times)[:chars]


def make_history(reasoning_chars: int = 12000, *, with_tools: bool = True) -> list[dict]:
    reasoning = long_reasoning(reasoning_chars)
    head, tail = reasoning[: len(reasoning) // 2], reasoning[len(reasoning) // 2:]
    messages: list[dict] = [{"role": "user", "content": "看一下订单 A1，决定要不要催单"}]
    if with_tools:
        messages += [
            {
                "role": "assistant",
                "content": "先查节点。",
                "reasoning_content": head,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "logistics.query", "arguments": '{"order_id":"A1"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": TOOL_A},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": tail,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "todo.write", "arguments": '{"list":"logistics"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": TOOL_B},
            {"role": "assistant", "content": "已经超时，建议催单。", "reasoning_content": "确认一次。"},
        ]
    else:
        messages += [
            {"role": "assistant", "content": "先想一下。", "reasoning_content": head},
            {"role": "assistant", "content": "结论是催单。", "reasoning_content": tail},
        ]
    messages.append({"role": "user", "content": "继续"})
    return messages


def good_llm(_prompt: str) -> str:
    return ("我在核对 A1 的物流节点：承诺 48 小时，实际 72 小时，差值 24 小时；"
            "等等，先别急着催，会不会是用户改了地址？历史里没有改地址记录，排除。"
            "那就按超时 24 小时处理，调用 express.urge 催单，备注写运单号 SF1234567890。"
            "但是转运中心停了 9 小时，还没到 24 小时的升级线，先不开异常件工单。") * 25


@pytest.fixture
def cfg(tmp_path) -> Config:
    # ratio_max 调大，避免测试被"压缩太狠"这一条挡掉；单独用例再考它
    return Config(shadow_dir=tmp_path / "shadow", keep_rounds=0, target_ratio_max=1000.0)


def test_below_threshold_not_cut(cfg):
    messages = make_history(3000)
    span, reason = select(messages, cfg)
    assert span is None
    assert reason == "reasoning_below_threshold"

    result = compact_messages(messages, cfg=cfg, llm_call=good_llm)
    assert result.compacted is False
    assert result.stats.reason == "reasoning_below_threshold"
    # 原样返回，不改数组长度，不写备份
    assert result.messages is messages
    assert not (cfg.shadow_dir / "spine-cuts.jsonl").exists()


def test_round_with_tool_calls_is_cut(cfg):
    messages = make_history()
    before = total_tokens(messages)
    span = select(messages, cfg)[0]
    assert span is not None
    assert [m["role"] for m in messages[span.start:span.end + 1]] == [
        "assistant", "tool", "assistant", "tool", "assistant",
    ]

    result = compact_messages(messages, cfg=cfg, llm_call=good_llm)
    assert result.compacted is True
    assert result.stats.reason == "ok"
    assert result.stats.tokens_after < before

    # 整段换成一条 user 卡片
    assert len(result.messages) == len(messages) - (span.end - span.start)
    card = result.messages[span.start]
    assert card["role"] == "user"
    assert card["checkpoint"] is True
    assert "reasoning_content" not in card and "tool_calls" not in card
    assert "[思考]" in card["content"]
    assert "[工具调用]" in card["content"]
    assert "[工具结果原文]" in card["content"]
    assert "[原文指针]" in card["content"]
    assert "logistics.query" in card["content"] and "todo.write" in card["content"]

    # 工具结果一字不改，且配对信息（call_id）还在
    assert TOOL_A in card["content"]
    assert TOOL_B in card["content"]
    assert "call_id=call_1" in card["content"] and "call_id=call_2" in card["content"]

    # 被切段里的 assistant/tool 一条都不剩
    assert not any(m.get("role") == "tool" for m in result.messages)


def test_backup_failure_keeps_original(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    cfg = Config(shadow_dir=blocked, keep_rounds=0, target_ratio_max=1000.0)
    messages = make_history()

    result = compact_messages(messages, cfg=cfg, llm_call=good_llm)
    assert result.compacted is False
    assert result.stats.reason == "backup_failed"
    assert result.messages is messages
    assert blocked.is_file()  # 备份目录没被"修好"，就是失败的


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        '{"stages":[{"facts":[],"reactions":[],"decisions":[]}]}',
        "[1,2,3]",
        None,
    ],
)
def test_invalid_summary_not_cut(cfg, bad):
    messages = make_history()
    result = compact_messages(messages, cfg=cfg, llm_call=lambda _p: bad)
    assert result.compacted is False
    assert result.stats.reason == "summary_invalid"
    assert result.messages is messages
    assert not any(m.get("checkpoint") for m in result.messages)


def test_over_compressed_summary_rejected(tmp_path):
    # 压缩比上限设成 1.0：摘要只要比原文短一倍以上就判不合格
    cfg = Config(shadow_dir=tmp_path / "shadow", keep_rounds=0, target_ratio_max=1.0)
    result = compact_messages(make_history(), cfg=cfg, llm_call=good_llm)
    assert result.compacted is False
    assert result.stats.reason == "summary_invalid"


def test_prefix_and_tail_objects_unchanged(cfg):
    messages = make_history()
    span = select(messages, cfg)[0]
    result = compact_messages(messages, cfg=cfg, llm_call=good_llm)

    # 切点之前：对象身份逐个相同（就是 prompt cache 命中要的"一个字节都不改"）
    for i in range(span.start):
        assert result.messages[i] is messages[i]
    # 切点之后：对象身份逐个相同，只是位置前移
    tail = messages[span.end + 1:]
    for i, message in enumerate(tail):
        assert result.messages[span.start + 1 + i] is message


def test_concurrent_prefix_mutation_raises(cfg):
    messages = make_history()

    def mutating_llm(prompt: str) -> str:
        # 模拟压缩期间别的线程改写了前缀
        messages[0]["content"] = "被改过的用户消息"
        return good_llm(prompt)

    with pytest.raises(RuntimeError):
        compact_messages(messages, cfg=cfg, llm_call=mutating_llm)


def test_existing_card_is_floor(cfg):
    messages = make_history()
    result = compact_messages(messages, cfg=cfg, llm_call=good_llm)
    span, reason = select(result.messages, cfg)
    assert span is None
    assert reason == "card_on_the_right"


def test_keep_rounds_protects_recent_tool_rounds(cfg):
    guarded = replace(cfg, keep_rounds=3)
    result = compact_messages(make_history(), cfg=guarded, llm_call=good_llm)
    assert result.compacted is False
    assert result.stats.reason == "recent_tool_rounds_kept"

    # 只保 1 轮：切前一轮，后一轮的 tool_calls 与 tool 结果必须同时留下
    one_kept = replace(cfg, keep_rounds=1)
    messages = make_history()
    result = compact_messages(messages, cfg=one_kept, llm_call=good_llm)
    assert result.compacted is True
    roles = [m.get("role") for m in result.messages]
    assert "tool" in roles
    # 留下的 tool 结果一定是它自己的 assistant 也留下了
    tail = result.messages[result.stats.span[0] + 1:]
    assert [m.get("role") for m in tail] == ["assistant", "tool", "assistant", "user"]
    assert tail[0]["tool_calls"][0]["id"] == "call_2"
    assert tail[1]["tool_call_id"] == "call_2"
    # 第一轮的工具结果进了卡片，第二轮没进
    assert TOOL_A in result.card
    assert TOOL_B not in result.card


def test_shadow_files_and_cut_log(cfg):
    messages = make_history()
    result = compact_messages(messages, cfg=cfg, llm_call=good_llm)
    backup_path = cfg.shadow_dir / (result.stats.backup_id + ".txt")
    assert backup_path.exists()

    raw = backup_path.read_text(encoding="utf-8")
    assert "[role:reasoning]" in raw
    assert "[role:assistant]" in raw
    assert "[role:tool]" in raw
    assert TOOL_A in raw and TOOL_B in raw

    lines = cfg.cuts_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {
        "seq_start", "seq_end", "tokens_before", "tokens_after",
        "reasoning_tokens", "backup",
    }
    assert record["backup"] == result.stats.backup_id
    # 日志只记序号与 token，不落原文
    assert TOOL_A not in lines[0]


def test_summarizer_validate():
    assert validate("我在想：等等，先看看 A1 的 48 小时承诺。")
    assert not validate("")
    assert not validate("   ")
    assert not validate('{"stages": []}')
    assert not validate("[1,2]")
    assert not validate(None)
    assert not validate({"stages": []})


def test_card_marks_missing_tool_result():
    from dsh_thinking_compact.card import build_card
    from dsh_thinking_compact.selector import Span

    segment = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "思考",
            "tool_calls": [
                {"id": "c9", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    ]
    span = Span(start=1, end=1, reasoning_tokens=10, cut_reasoning_tokens=10, segment_tokens=10)
    card = build_card(
        span=span,
        segment=segment,
        summary="我在想：等等，先看看这个调用有没有结果。不对，段内没有匹配结果，得如实标注。",
        backup_id="sp-test",
        digest="0" * 64,
        nbytes=1,
    )
    # 调用没有对应结果时如实标注，不编造块编号
    assert "（切段内无匹配结果）" in card
    assert "sha256=" + "0" * 64 in card
    assert "seq 1..1" in card


def test_no_llm_call_does_nothing(cfg):
    messages = make_history()
    result = compact_messages(messages, cfg=cfg, llm_call=None)
    assert result.compacted is False
    assert result.stats.reason == "no_llm_call"
    assert result.messages is messages
