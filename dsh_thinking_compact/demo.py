"""离线演示：不联网、不读真实会话，构造一段假历史看完整流程。

跑法：python -m dsh_thinking_compact.demo
演示内容：
1. reasoning 不够阈值 -> 不切（原样返回）
2. keep_rounds=3 时会保住最近的工具轮（默认已改 0，这里显式演示保护）-> 单轮不切；keep_rounds=0 时才落刀
3. 一整轮（含 tool_calls + tool 结果）换成一张卡片
4. 工具结果原文在卡片里逐字保留
5. 备份文件真的落盘，spine-cuts.jsonl 有记录
6. 切段前缀/后缀的消息对象身份不变（prompt cache 的前提）
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from .apply import compact_messages, stats_dict
from .config import Config
from .selector import select

# 一段"像真的"的思考：带数值、边界、被否方案，用来演示压缩后这些都得留下
THINKING = (
    "先确认订单号 A1 的物流状态。查到的结果是 3 天前已发货，承运商是顺丰，"
    "单号 SF1234567890，最新节点是 2024-05-11 09:20 到达杭州转运中心。"
    "用户说'要不要催单'，所以要判断是否超过承诺时效。承诺是 48 小时内送达，"
    "现在从发货算起已经 72 小时，超过 24 小时，属于可以催的范围。"
    "但先别急，会不会是用户自己改了地址导致延迟？历史里没有改地址的记录。"
    "考虑两个方案：方案一是直接催单，方案二是先告诉用户现状让用户决定。"
    "方案二更稳，但如果用户已经明确要催，方案二就多一次往返。用户这次用的是"
    "'帮我决定'，说明授权我自己判断，所以选方案一。"
    "另一个被否掉的是'申请退款'，因为货已经在路上，退款需要拦截，成本更高。"
    "边界条件：如果转运中心超过 24 小时没有新节点，才升级为异常件工单。"
    "现在只过去了 9 小时，不触发升级。数值上再核一遍：72 - 48 = 24，没错。"
)

TODO_BODY = (
    "1. 调用 logistics.query 拿最新节点\n"
    "2. 对比承诺时效 48h\n"
    "3. 超过则调用 express.urge 催单，并把单号写进备注\n"
    "4. 回复用户：已催单 + 预计明日送达"
)

CARRIER_BODY = (
    "{\n"
    '  "carrier": "SF",\n'
    '  "tracking_no": "SF1234567890",\n'
    '  "events": [\n'
    '    {"time": "2024-05-11 09:20", "node": "杭州转运中心", "status": "in_transit"},\n'
    '    {"time": "2024-05-09 18:40", "node": "深圳仓", "status": "shipped"}\n'
    "  ],\n"
    '  "promise_hours": 48\n'
    "}"
)


def _pad(text: str, target_chars: int) -> str:
    """把一段思考拼到目标长度，模拟真实的超长 reasoning（约 3 字符/token）。"""
    if len(text) >= target_chars:
        return text[:target_chars]
    times = target_chars // len(text) + 1
    return (text * times)[:target_chars]


def build_conversation(reasoning_chars: int) -> list[dict]:
    reasoning = _pad(THINKING, reasoning_chars)
    head, tail = reasoning[: len(reasoning) // 2], reasoning[len(reasoning) // 2:]
    return [
        {"role": "user", "content": "帮我看看订单 A1 的物流，然后决定要不要催单"},
        {
            "role": "assistant",
            "content": "我先拉一下物流节点。",
            "reasoning_content": head,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "logistics.query", "arguments": '{"order_id":"A1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": CARRIER_BODY},
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
        {"role": "tool", "tool_call_id": "call_2", "content": TODO_BODY},
        {
            "role": "assistant",
            "content": "结论：已发货 72 小时，超过承诺 48 小时，建议催单。",
            "reasoning_content": "最后确认一次：用户授权我自己决定，方案一成立。",
        },
        {"role": "user", "content": "继续"},
    ]


def mock_llm(prompt: str) -> str:
    """假模型：不看 prompt，直接吐一段缩句后的思考，用来演示链路。

    内容刻意写得够长：比值要低于 target_ratio_max=8 才算合格摘要，压太狠本身就
    说明漏了东西；同时保留第一人称和转折口气。
    """
    assert "你在压缩一段模型的思考过程" in prompt
    assert "{TEXT}" not in prompt
    return (THINKING * 8)[:9000]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="thinking-compact-demo-") as tmp:
        base = Config(shadow_dir=Path(tmp) / "shadow")
        print("== 配置 ==")
        print(
            "min_reasoning_tokens=%d chunk_min_tokens=%d keep_rounds=%d max_context=%d target_ratio_max=%s"
            % (
                base.min_reasoning_tokens,
                base.chunk_min_tokens,
                base.keep_rounds,
                base.max_context_tokens,
                base.target_ratio_max,
            )
        )
        print("shadow_dir=%s" % base.shadow_dir)

        # ---- 场景 1：reasoning 不到阈值 ----
        print("\n== 场景 1：reasoning 约 3000 字符（≈1000 token），不到 4000 ==")
        short = build_conversation(3000)
        span, reason = select(short, base)
        result = compact_messages(short, cfg=base, llm_call=mock_llm)
        print("选段:", span, "| 原因:", reason)
        print("是否压缩:", result.compacted, "| 原因:", result.stats.reason,
              "| 返回的就是原数组:", result.messages is short)

        # ---- 场景 2a：显式 keep_rounds=3 保护最近的工具轮 ----
        print("\n== 场景 2a：reasoning 约 12000 字符（≈4000 token），显式 keep_rounds=3 ==")
        guarded_cfg = replace(base, keep_rounds=3)
        guarded = compact_messages(build_conversation(12000), cfg=guarded_cfg, llm_call=mock_llm)
        print("是否压缩:", guarded.compacted, "| 原因:", guarded.stats.reason)
        print("说明：这一轮只有 2 个工具轮，全在'最近 3 轮留原文'的保护范围内。")

        # ---- 场景 2b：keep_rounds=0，真正落刀 ----
        cfg = replace(base, keep_rounds=0)
        print("\n== 场景 2b：keep_rounds=0，同一段历史应被整段替换 ==")
        messages = build_conversation(12000)
        span, reason = select(messages, cfg)
        print("选段:", (span.start, span.end) if span else None, "| 原因:", reason,
              "| 切段 token:", span.segment_tokens if span else 0)
        print("被切消息角色:", [m["role"] for m in messages[span.start:span.end + 1]])

        result = compact_messages(messages, cfg=cfg, llm_call=mock_llm)
        stats = result.stats
        print("是否压缩:", result.compacted)
        print("统计:", json.dumps(stats_dict(stats), ensure_ascii=False))
        print("压缩前 token ≈ %d -> 压缩后 ≈ %d（省 %d）"
              % (stats.tokens_before, stats.tokens_after, stats.saved_tokens))
        print("替换后消息角色:", [m["role"] for m in result.messages])
        print("说明：卡片是一条 user 消息，占在被切轮次原位，所以会夹在两条 user 之间"
              "（与 spine.py 的落点一致）。")

        # ---- 硬校验 ----
        start, end = stats.span
        card_message = result.messages[start]
        prefix_same = all(result.messages[i] is messages[i] for i in range(start))
        tail_same = all(
            result.messages[start + 1 + i] is m for i, m in enumerate(messages[end + 1:])
        )
        tool_bodies = [m["content"] for m in messages if m.get("role") == "tool"]
        tool_ok = all(body in result.card for body in tool_bodies)
        backup = cfg.shadow_dir / (stats.backup_id + ".txt")
        cuts = cfg.cuts_path
        print("卡片角色/标记:", card_message["role"], card_message.get("checkpoint"),
              "| 卡片不带 reasoning/tool_calls:",
              "reasoning_content" not in card_message and "tool_calls" not in card_message)
        print("前缀对象未变（prompt cache 前提）:", prefix_same)
        print("后缀对象未变:", tail_same)
        print("工具结果原文完整保留:", tool_ok)
        print("备份文件存在:", backup.exists(), "| 字节:",
              backup.stat().st_size if backup.exists() else 0)
        print("spine-cuts.jsonl 行数:",
              len(cuts.read_text(encoding="utf-8").splitlines()) if cuts.exists() else 0)
        print("卡片里没有本机路径:", tmp not in result.card)

        # ---- 卡片全文 ----
        print("\n== 卡片全文 ==")
        print(result.card)

        # ---- 场景 3：摘要不合格 -> 不压缩 ----
        print("\n== 场景 3：模型输出不合格（空文本），整段保留、不压缩 ==")
        bad = compact_messages(build_conversation(12000), cfg=cfg,
                               llm_call=lambda _p: "")
        print("是否压缩:", bad.compacted, "| 原因:", bad.stats.reason,
              "| 消息条数仍为:", len(bad.messages))

        # ---- 场景 4：已有卡片，再选段 ----
        print("\n== 场景 4：卡片已是地板，再选段 ==")
        again, why = select(result.messages, cfg)
        print("选段:", again, "| 原因:", why)

        # ---- 场景 5：按会话驻留 ----
        print("\n== 场景 5：驻留视图（每次只压新追加的增量，卡片固化）==")
        from .session import SessionStore

        store = SessionStore(cfg)
        calls: list[str] = []

        def counting_llm(prompt: str) -> str:
            calls.append(prompt)
            return mock_llm(prompt)

        def anthropic_round(index: int) -> list[dict]:
            call_id = "call_%d" % index
            return [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": _pad(THINKING, 12000)},
                    {"type": "text", "text": "先查节点。"},
                    {"type": "tool_use", "id": call_id, "name": "logistics.query",
                     "input": {"order_id": "A1"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": CARRIER_BODY},
                ]},
            ]

        session = [{"role": "user", "content": [{"type": "text", "text": "帮我看看订单 A1"}]}]
        views: list[list[dict]] = []
        for step in range(1, 5):
            session = session + anthropic_round(step)
            view, st = store.render(
                {"system": "你是调度助理", "messages": list(session)}, counting_llm
            )
            views.append(view)
            print("第 %d 次请求：原文 %d 条 -> 发出 %d 条 | 本轮切段=%s | 累计卡片=%d | token ≈ %d -> %d"
                  % (step, len(session), len(view), st.get("compacted"), st.get("cards"),
                     st["tokens_before"], st["tokens_after"]))

        append_only = all(views[i + 1][:len(views[i])] == views[i] for i in range(len(views) - 1))
        # 旧实现每请求都把完整历史重压一遍；这里只算思考字符总量做对比
        old_chars = 0
        acc = [{"role": "user", "content": [{"type": "text", "text": "帮我看看订单 A1"}]}]
        for step in range(1, 5):
            acc = acc + anthropic_round(step)
            old_chars += sum(
                len(block.get("thinking") or "")
                for message in acc if isinstance(message.get("content"), list)
                for block in message["content"]
                if isinstance(block, dict) and block.get("type") == "thinking"
            )
        print("摘要调用次数:", len(calls), "| 每次输入字符数:", [len(p) for p in calls])
        print("旧实现（每请求重压整段）摘要输入合计 ≈ %d 字符，驻留实现 ≈ %d 字符（少 %d%%）"
              % (old_chars, sum(len(p) for p in calls),
                 round(100 * (1 - sum(len(p) for p in calls) / max(1, old_chars)))))
        print("视图严格只追加（已发出的前缀永不回改）:", append_only)
        _view, again_stats = store.render(
            {"system": "你是调度助理", "messages": list(session)}, counting_llm
        )
        print("同一原文重复发送：本轮切段=%s | 原因=%s | 摘要调用总数=%d"
              % (again_stats.get("compacted"), again_stats.get("reason"), len(calls)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
