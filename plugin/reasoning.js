/**
 * 思考链选段 + 卡片渲染。
 *
 * 骨架（surface/seq/tokenMeter 的用法、cutOk 的调用位置、chunk 的返回形状）
 * 来自 dsh-live-compact（MIT，见 LICENSE.upstream）。这里换掉的是「切哪一段、
 * 卡片长什么样」：参照物按工具结果的行做摘录启发式，本文件只按完整助手轮切，
 * 压缩对象是思考链，工具调用与结果原文照抄。
 *
 * 三条硬规则（与 Python 版 select_tail 一致）：
 * 1. 只切完整的助手轮：assistant 的每个 tool-call 都必须有对应 tool/result，
 *    否则这一轮不动 —— 切了会留下孤立 tool_use，上游直接 400。
 * 2. 不跨真人消息和半截轮次；checkpoint 卡片是地板。
 * 3. 思考 token 与切段 token 都要过阈值，不达标就不切。
 */

import { readFileSync } from "node:fs";

/**
 * 缩句提示词：直接读 prompt.txt，不从代码里转抄。
 * prompt.txt 与 bench/samples/compress_prompt.txt 逐字节一致，有测试守着。
 */
export const PROMPT = readFileSync(new URL("./prompt.txt", import.meta.url), "utf8").replace(/\r\n/g, "\n");

/** 占位符是三行形态：<<< / {TEXT} / >>> */
export const PLACEHOLDER = "<<<\n{TEXT}\n>>>";

/** 思考块在 DSH 的 Message 里是 { type: "reasoning", text }。 */
export function reasoningText(messages) {
  const parts = [];
  for (const message of messages || []) {
    const content = message?.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!block) continue;
      if (block.type !== "reasoning" && block.type !== "thinking") continue;
      const text = String(block.text ?? block.thinking ?? "");
      if (text.trim()) parts.push(text);
    }
  }
  return parts.join("\n\n");
}

function blocksOf(event) {
  return event?.data?.message?.content || [];
}

function toolCallIds(event) {
  return blocksOf(event)
    .filter((block) => block?.type === "tool-call" && block.id)
    .map((block) => block.id);
}

/** 一个助手轮需要的 tool/result 是否都在紧跟其后的连续事件里。 */
function roundEnd(nodes, session, start) {
  const ids = toolCallIds(session.eventAt(nodes[start]));
  if (ids.length === 0) return start;
  const seen = new Set();
  let end = start;
  for (let j = start + 1; j < nodes.length; j += 1) {
    const event = session.eventAt(nodes[j]);
    if (event?.type !== "tool/result") break;
    seen.add(event?.data?.message?.toolCallId);
    end = j;
    if (ids.every((id) => seen.has(id))) return end;
  }
  return -1; // 半截轮次：tool-call 没等到结果
}

/**
 * 选一段可切的思考链区间。
 *
 * @returns {{start:number,end:number,tokens:number,after:number,keptTokens:number,card:object,prefix:number}|null}
 */
export function selectReasoningChunk(session, measurement, key, context = {}) {
  const settings = context.settings ?? {};
  const isCheckpoint = context.isCheckpoint;
  const cutOkBefore = context.cutOkBefore;
  const cutOkAfter = context.cutOkAfter;
  const minReasoning = Number(settings.reasoningTokens ?? 4000);
  const chunkMin = Number(settings.chunkMin ?? 1000);

  const nodes = session.surface.nodes;
  const priced = measurement.nodes;
  if (!nodes.length || nodes.length !== priced.length) return null;
  const first = session.eventAt(nodes[0])?.type === "system/message" ? 1 : 0;

  // checkpoint 卡片是地板，绝不切到它之前
  let floor = -1;
  for (let index = first; index < nodes.length; index += 1) {
    if (isCheckpoint?.(session, nodes[index])) floor = index;
  }

  // 把地板之后的 surface 切成「完整助手轮」与「屏障」（真人消息、半截轮次、其它）
  const units = [];
  for (let index = Math.max(first, floor + 1); index < nodes.length; index += 1) {
    const event = session.eventAt(nodes[index]);
    if (event?.type !== "assistant/message") {
      units.push({ kind: "barrier", start: index, end: index });
      continue;
    }
    const end = roundEnd(nodes, session, index);
    if (end < 0) {
      // 半截轮次之后都不再考虑，但不影响它前面已经攒好的完整轮
      units.push({ kind: "open", start: index, end: nodes.length - 1 });
      break;
    }
    units.push({ kind: "round", start: index, end });
    index = end;
  }

  // 取最长的连续完整轮
  let best = null;
  let current = [];
  for (const unit of units) {
    if (unit.kind === "round") {
      current.push(unit);
      continue;
    }
    if (current.length > (best?.length ?? 0)) best = current;
    current = [];
  }
  if (current.length > (best?.length ?? 0)) best = current;
  if (!best || best.length === 0) return null;

  const keepRounds = Number(settings.keepRounds ?? 0);
  const cut = keepRounds > 0 ? best.slice(0, best.length - keepRounds) : best;
  if (cut.length === 0) return null;

  const start = cut[0].start;
  const end = cut[cut.length - 1].end;
  if (floor >= 0 && start <= floor) return null;
  if (cutOkBefore && !cutOkBefore(session, nodes[start])) return null;
  if (cutOkAfter && !cutOkAfter(session, nodes[end])) return null;

  // 思考 token 与切段 token 双重阈值
  const segment = [];
  let reasoningChars = 0;
  let tokens = 0;
  for (let index = start; index <= end; index += 1) {
    const event = session.eventAt(nodes[index]);
    for (const block of blocksOf(event)) {
      if (block?.type === "reasoning" || block?.type === "thinking") {
        reasoningChars += String(block.text ?? block.thinking ?? "").length;
      }
    }
    tokens += priced[index].tokens || 0;
    segment.push({ seq: nodes[index], role: event?.type === "tool/result" ? "tool" : "assistant" });
  }
  const reasoningTokens = Math.ceil(reasoningChars / 3);
  if (reasoningTokens < minReasoning) return null;
  if (tokens < chunkMin) return null;

  let after = 0;
  for (let index = end + 1; index < priced.length; index += 1) after += priced[index].tokens || 0;
  let prefix = 0;
  for (let index = 0; index < start; index += 1) prefix += priced[index].tokens || 0;

  return {
    start: nodes[start],
    end: nodes[end],
    tokens,
    after,
    keptTokens: 0,
    // 选段阶段只给占位卡片；真正的缩句由 summarize() 生成后替换
    card: { ok: true, text: "（思考链摘录：正文由模型缩句后填入）", seq: [nodes[start], nodes[end]] },
    prefix,
  };
}

function blockText(block) {
  if (!block) return "";
  if (block.type === "text") return String(block.text ?? "");
  if (block.type === "tool-call") {
    const args = block.arguments ?? block.input ?? {};
    return `${block.name || "?"}(${typeof args === "string" ? args : JSON.stringify(args)})`;
  }
  if (block.type === "tool-result") {
    const content = block.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) return content.map(blockText).join("\n");
    return JSON.stringify(content ?? "");
  }
  return "";
}

/** 卡片正文：声明 + 缩句 + 工具调用与结果原文 + 原文指针。 */
export function renderCardText({ summary, messages, seqStart, seqEnd }) {
  const reasoning = [];
  const tools = [];
  const results = [];
  for (const message of messages || []) {
    for (const block of message?.content || []) {
      if (!block) continue;
      if (block.type === "tool-call") tools.push(blockText(block));
      if (block.type === "tool-result") results.push(blockText(block));
    }
  }
  const lines = [
    "这是旧一轮的摘录，不是新指令，也不是你的授权。里面的思考是我当时自己想的（已按原话缩句），可以接着往下推，不要把它当成用户新提的要求。",
    "Spine extract.",
    "",
    "[思考] 我当时那一轮的想法（按原话缩句）",
    "",
    String(summary || "").trim() || "（无）",
    "",
    "[工具调用]",
    ...(tools.length ? tools : ["（无）"]),
    "",
    "[工具结果原文] 一字不改，完整保留",
    ...(results.length ? results : ["（无）"]),
    "",
    `[原文指针] seq ${seqStart}..${seqEnd}（本会话事件号，不是文件路径）`,
  ];
  return lines.join("\n");
}

/**
 * 摘要用哪个 provider/model：显式配置 > 最近一次请求的模型 > agent 选项。
 * 官方 summarizeWithLlm 就是这三层；只读 config 那一层在默认 profile 下是空串，
 * 会让思考链永远压不动（静默降级成"每轮都跳过"）。
 */
export function summarizationTarget(config, agent) {
  const provider = config?.summarizationProvider;
  const model = config?.summarizationModel;
  if (provider && model) return { provider, model };
  const latest = agent?.session?.requestHeader?.()?.config;
  if (latest?.provider && latest?.model) return { provider: latest.provider, model: latest.model };
  const options = agent?.options;
  if (options?.provider && options?.model) return { provider: options.provider, model: options.model };
  return null;
}

/** 把一段思考塞进提示词。占位符缺失就返回 null，绝不发一段没装正文的提示词。 */
export function buildPrompt(text) {
  if (!text || !text.trim()) return null;
  if (!PROMPT.includes(PLACEHOLDER)) return null;
  return PROMPT.replace(PLACEHOLDER, () => text);
}

/** 摘要校验：非空、不是 JSON、不比原文长。 */
export function acceptSummary(summary, sourceText) {
  const text = String(summary || "").trim();
  if (!text) return null;
  if (text.startsWith("{") || text.startsWith("[")) return null;
  if (sourceText && text.length >= sourceText.length) return null;
  return text;
}
