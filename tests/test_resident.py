"""驻留视图的验收测试。

核心不变式只有三条：
1. 压缩只发生在"还没发出去过"的增量上，已发出的前缀永不回改；
2. 同一段原文只压一次，卡片生成后就固化，后续请求原样复用；
3. 原文前缀变了就截断重建，绝不拿旧卡片去套新原文。
"""
from __future__ import annotations

from dsh_thinking_compact.config import Config
from dsh_thinking_compact.messages import select_tail
from dsh_thinking_compact.session import SessionStore, fingerprint_of

UNIT = "核对 A1 的物流节点，承诺 48 小时，实际 72 小时，差值 24 小时；"

SUMMARY = (
    "我在核对 A1 的物流节点：承诺 48 小时，实际 72 小时，差值 24 小时。"
    "等等，先别急着催，会不会是用户改了地址？历史里没有改地址记录，排除。"
    "那就按超时 24 小时处理，调 express.urge 催单，备注写运单号 SF1234567890。"
    "但是转运中心停了 9 小时，还没到 24 小时升级线，先不开异常件工单。"
)


def reasoning(chars: int = 12000) -> str:
    return (UNIT * (chars // len(UNIT) + 1))[:chars]


def tool_round(index: int, chars: int = 12000) -> list[dict]:
    call_id = "call_%d" % index
    return [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning(chars)},
            {"type": "text", "text": "先查节点。"},
            {"type": "tool_use", "id": call_id, "name": "logistics.query",
             "input": {"order_id": "A1"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": "节点：已到转运中心"},
        ]},
    ]


def history(rounds: int, chars: int = 12000) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": "看一下订单 A1"}]}]
    for i in range(rounds):
        messages += tool_round(i, chars)
    return messages


def make_store(tmp_path) -> SessionStore:
    cfg = Config(shadow_dir=tmp_path / "shadow", keep_rounds=0, target_ratio_max=1000.0)
    return SessionStore(cfg)


def payload(rounds: int, system: str = "你是调度助理", chars: int = 12000) -> dict:
    return {"system": system, "messages": history(rounds, chars)}


def test_only_new_tail_is_cut(tmp_path):
    store = make_store(tmp_path)
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return SUMMARY

    views = []
    for rounds in (1, 2, 3):
        view, stats = store.render(payload(rounds), llm)
        assert stats["compacted"] is True, stats
        views.append(view)

    # 三次请求各切一刀，而不是每次把整段历史重压一遍
    assert len(calls) == 3
    # 视图长度：1 human + 每轮一张卡
    assert [len(v) for v in views] == [2, 3, 4]


def test_view_is_append_only(tmp_path):
    store = make_store(tmp_path)
    views = []
    for rounds in (1, 2, 3, 4):
        view, _ = store.render(payload(rounds), lambda _p: SUMMARY)
        views.append(view)

    for earlier, later in zip(views, views[1:]):
        # 上一次发出去的字节必须原样留在这一次的开头
        assert later[:len(earlier)] == earlier


def test_same_payload_is_not_cut_twice(tmp_path):
    store = make_store(tmp_path)
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return SUMMARY

    first, stats1 = store.render(payload(2), llm)
    second, stats2 = store.render(payload(2), llm)

    assert stats1["compacted"] is True
    assert stats2["compacted"] is False
    assert stats2["reason"] == "nothing_new"
    assert len(calls) == 1
    assert first == second


def test_card_text_is_stable(tmp_path):
    store = make_store(tmp_path)
    view_a, _ = store.render(payload(2), lambda _p: SUMMARY)
    card_a = view_a[-1]["content"][0]["text"]
    # 再发一次更长的历史，旧卡片必须逐字不变
    view_b, _ = store.render(payload(3), lambda _p: SUMMARY)
    assert view_b[len(view_a) - 1]["content"][0]["text"] == card_a


def test_prefix_change_truncates_and_rebuilds(tmp_path):
    store = make_store(tmp_path)
    store.render(payload(2), lambda _p: SUMMARY)

    mutated = payload(3)
    mutated["messages"][0] = {"role": "user", "content": [{"type": "text", "text": "换个订单"}]}
    view, stats = store.render(mutated, lambda _p: SUMMARY)

    # 前缀对不上：视图被清空重来，第一项必须是新的第一条原文
    assert view[0] == mutated["messages"][0]
    assert stats["absorbed_before"] == 0


def test_summary_failure_absorbs_verbatim(tmp_path):
    store = make_store(tmp_path)
    view, stats = store.render(payload(1), lambda _p: "")
    assert stats["compacted"] is False
    assert stats["reason"] in ("summary_invalid", "segment_too_small")
    # 原样吸收，不半途改
    assert view == payload(1)["messages"]


def test_fingerprint_separates_sessions(tmp_path):
    store = make_store(tmp_path)
    a = store.render(payload(2, system="会话 A"), lambda _p: SUMMARY)
    # B 组的思考短到过不了阈值，不会被切，正好用来验证它没被 A 的视图污染
    b_input = payload(2, system="会话 B", chars=3000)
    b = store.render(b_input, lambda _p: SUMMARY)
    assert a[1]["fingerprint"] != b[1]["fingerprint"]
    assert b[1]["compacted"] is False
    assert b[0] == b_input["messages"]


def test_select_tail_stops_at_open_round():
    tail = tool_round(0) + [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": reasoning()},
        {"type": "tool_use", "id": "call_x", "name": "todo.write", "input": {}},
    ]}]
    span, reason = select_tail(tail)
    # 只切前面那个完整轮，半截轮次不动
    assert reason == "ok"
    assert (span.start, span.end) == (0, 1)


def test_select_tail_refuses_incomplete_only():
    tail = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": reasoning()},
        {"type": "tool_use", "id": "call_x", "name": "todo.write", "input": {}},
    ]}]
    span, reason = select_tail(tail)
    assert span is None
    assert reason == "no_closed_round"


def test_fingerprint_stable_across_growth():
    short = history(1)
    long = history(3)
    assert fingerprint_of("sys", short) == fingerprint_of("sys", long)
