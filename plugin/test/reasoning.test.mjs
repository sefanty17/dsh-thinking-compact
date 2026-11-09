/**
 * 选段与卡片的离线验收。不需要 DSH 运行时：session / measurement 用假对象喂进去。
 *
 * 跑法：node --test plugin/test
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptSummary,
  buildPrompt,
  reasoningText,
  renderCardText,
  selectReasoningChunk,
} from "../reasoning.js";

const UNIT = "核对 A1 的物流节点，承诺 48 小时，实际 72 小时，差值 24 小时；";

function reasoning(chars = 12000) {
  return (UNIT.repeat(Math.ceil(chars / UNIT.length))).slice(0, chars);
}

/** 造一个假 session：nodes 是事件号，eventAt 返回 DSH 形状的事件。 */
function makeSession(events) {
  const nodes = events.map((event, index) => 100 + index);
  const bySeq = new Map(nodes.map((seq, index) => [seq, events[index]]));
  return {
    surface: { nodes },
    eventAt: (seq) => bySeq.get(seq),
  };
}

function assistant(text, toolCalls = [], think = reasoning()) {
  const content = [];
  if (think) content.push({ type: "reasoning", text: think });
  if (text) content.push({ type: "text", text });
  for (const id of toolCalls) {
    content.push({ type: "tool-call", id, name: "logistics.query", arguments: { order_id: "A1" } });
  }
  return { type: "assistant/message", data: { message: { content } } };
}

function toolResult(id, body = "节点：已到转运中心") {
  return { type: "tool/result", data: { message: { toolCallId: id, content: [{ type: "text", text: body }] } } };
}

function human(text = "看一下订单 A1") {
  return { type: "user/message", data: { message: { content: [{ type: "text", text }] } } };
}

function measurement(session, tokensPerEvent = 2000) {
  const nodes = session.surface.nodes.map(() => ({ tokens: tokensPerEvent }));
  const totalTokens = nodes.reduce((sum, node) => sum + node.tokens, 0);
  return { nodes, totalTokens };
}

const noCut = () => true;

test("完整两轮会被整段选中", () => {
  const session = makeSession([
    human(),
    assistant("先查节点。", ["c1"]),
    toolResult("c1"),
    assistant("", ["c2"]),
    toolResult("c2"),
  ]);
  const chunk = selectReasoningChunk(session, measurement(session), "k", {
    settings: { minReasoning: 4000, chunkMin: 1000 },
    isCheckpoint: () => false,
    cutOkBefore: noCut,
    cutOkAfter: noCut,
  });
  assert.ok(chunk, "应该切中");
  assert.equal(chunk.start, session.surface.nodes[1]);
  assert.equal(chunk.end, session.surface.nodes[4]);
  assert.equal(chunk.keptTokens, 0);
  assert.ok(chunk.card.ok);
});

test("半截轮次（tool-call 没有结果）不切", () => {
  const session = makeSession([human(), assistant("", ["c1"])]);
  const chunk = selectReasoningChunk(session, measurement(session), "k", {
    settings: {},
    isCheckpoint: () => false,
    cutOkBefore: noCut,
    cutOkAfter: noCut,
  });
  assert.equal(chunk, null);
});

test("思考不够阈值不切", () => {
  const session = makeSession([
    human(),
    assistant("先查节点。", ["c1"], reasoning(3000)), // ≈1000 token
    toolResult("c1"),
  ]);
  const chunk = selectReasoningChunk(session, measurement(session), "k", {
    settings: { reasoningTokens: 4000 },
    isCheckpoint: () => false,
    cutOkBefore: noCut,
    cutOkAfter: noCut,
  });
  assert.equal(chunk, null);
});

test("checkpoint 卡片是地板，切不到它前面", () => {
  const session = makeSession([
    human(),
    assistant("", ["c1"]),
    toolResult("c1"),
    { type: "user/message", data: { message: { content: [{ type: "text", text: "Spine extract." }] } } },
  ]);
  const checkpointSeq = session.surface.nodes[3];
  const chunk = selectReasoningChunk(session, measurement(session), "k", {
    settings: {},
    isCheckpoint: (_session, seq) => seq === checkpointSeq,
    cutOkBefore: noCut,
    cutOkAfter: noCut,
  });
  assert.equal(chunk, null, "地板之后没有完整轮，不该切");
});

test("工具配对不平衡就放弃", () => {
  const session = makeSession([
    human(),
    assistant("", ["c1"]),
    toolResult("c1"),
  ]);
  const chunk = selectReasoningChunk(session, measurement(session), "k", {
    settings: {},
    isCheckpoint: () => false,
    cutOkBefore: noCut,
    cutOkAfter: () => false,
  });
  assert.equal(chunk, null);
});

test("卡片里工具原文逐字保留，且声明不是新指令", () => {
  const messages = [
    { role: "assistant", content: [
      { type: "reasoning", text: "想了很久" },
      { type: "tool-call", id: "c1", name: "logistics.query", arguments: { order_id: "A1" } },
    ] },
    { role: "user", content: [{ type: "tool-result", content: "节点：已到转运中心" }] },
  ];
  const text = renderCardText({
    summary: "我在核对 A1 的物流节点。",
    messages,
    seqStart: 101,
    seqEnd: 102,
  });
  assert.match(text, /不是新指令/);
  assert.match(text, /我在核对 A1 的物流节点。/);
  assert.match(text, /logistics\.query/);
  assert.match(text, /节点：已到转运中心/);
  assert.match(text, /seq 101\.\.102/);
});

test("摘要校验：空 / JSON / 比原文长 都拒", () => {
  assert.equal(acceptSummary("", "原文"), null);
  assert.equal(acceptSummary('{"a":1}', "原文"), null);
  assert.equal(acceptSummary("很长".repeat(50), "短"), null);
  assert.equal(acceptSummary("像我自己想的缩句。", "原文很长".repeat(20)), "像我自己想的缩句。");
});

test("提示词占位符能替换，且思考块能抽出来", () => {
  const text = reasoningText([{ content: [{ type: "reasoning", text: "abc" }] }]);
  assert.equal(text, "abc");
  const prompt = buildPrompt("要压的东西");
  assert.ok(prompt.includes("要压的东西"));
  assert.ok(!prompt.includes("<<<{TEXT}>>>"));
});
