import { createRequire } from "node:module";
import { STOCK_ENGINE as STOCK } from "./adapter.js";
import { configSnapshot, engineConfig } from "./settings.js";

const OURS = "dsh-live-compact";
const PRESET_IDS = new Set(["standard", "ptc", "minimal", "cordis", "preset-standard", "preset-ptc", "preset-minimal", "preset-cordis"]);
export const GATE_SERVICE = "liveCompactPresetGate";
export const PRESET_ENGINE_CONFIG = Object.freeze(engineConfig());

const require = createRequire(import.meta.url);

export function packageLoadable(load = (name) => require(name)) {
  try {
    load(OURS);
    return true;
  } catch (error) {
    // The shell uses top-level await so a failed engine does not fail this package.
    // require() refuses that shape even when import() can load it.
    if (error?.code !== "ERR_REQUIRE_ASYNC_MODULE" && error?.code !== "ERR_REQUIRE_ESM") return false;
    try {
      require.resolve(OURS);
      return true;
    } catch {
      return false;
    }
  }
}

function isPreset(node) {
  if (!node || typeof node !== "object") return false;
  if (PRESET_IDS.has(node.id)) return true;
  return Boolean(node.config && typeof node.config === "object" && !Array.isArray(node.config) && PRESET_IDS.has(node.config.id));
}

function withEngineConfig(entry, rootConfig) {
  const previous = entry.config && typeof entry.config === "object" && !Array.isArray(entry.config) ? configSnapshot(entry.config) : {};
  const { takeoverPresets: _takeover, ...root } = configSnapshot(rootConfig);
  const merged = { ...root, ...previous };
  // 较高优先级显式选择 retention 模式时，移除继承的互斥字段。
  if (previous.retainRatio !== undefined && previous.retainTokens === undefined) delete merged.retainTokens;
  if (previous.retainTokens !== undefined && previous.retainRatio === undefined) delete merged.retainRatio;
  return { ...merged, ...engineConfig(merged) };
}

function rewriteEntry(entry, loadable, rootConfig) {
  if (!entry || typeof entry !== "object") return entry;
  const next = { ...entry };
  if (entry.name === STOCK) {
    if (!loadable) return entry;
    next.name = OURS;
    next.config = withEngineConfig(entry, rootConfig);
    return next;
  }
  if (Array.isArray(entry.config)) next.config = entry.config.map((item) => rewriteEntry(item, loadable, rootConfig));
  else if (entry.config && typeof entry.config === "object") next.config = rewriteNode(entry.config, loadable, rootConfig);
  return next;
}

function rewriteNode(node, loadable, rootConfig) {
  if (Array.isArray(node)) return node.map((item) => rewriteEntry(item, loadable, rootConfig));
  if (!node || typeof node !== "object") return node;
  const next = { ...node };
  if (Array.isArray(node.plugins)) next.plugins = node.plugins.map((item) => rewriteEntry(item, loadable, rootConfig));
  if (Array.isArray(node.config)) next.config = node.config.map((item) => rewriteEntry(item, loadable, rootConfig));
  else if (node.config && typeof node.config === "object") next.config = rewriteNode(node.config, loadable, rootConfig);
  return next;
}

export function rewritePresetConfig(config, loadable = packageLoadable(), rootConfig = {}) {
  if (!isPreset(config)) return config;
  return rewriteNode(config, loadable, rootConfig);
}

function engineEntry(node) {
  if (!node || typeof node !== "object") return null;
  if (node.name === OURS) return node;
  const nested = [];
  if (Array.isArray(node.plugins)) nested.push(...node.plugins);
  if (Array.isArray(node.config)) nested.push(...node.config);
  else if (node.config && typeof node.config === "object") nested.push(node.config);
  for (const item of nested) {
    const found = engineEntry(item);
    if (found) return found;
  }
  return null;
}

// A compatibility gate is provisional. The shallowest real entry owns the one
// listener; retain its authority while unloaded so nested/legacy reloads cannot
// silently enable takeover again.
const installedGates = new WeakMap();

function entryDepth(ctx) {
  let depth = 0;
  for (let fiber = ctx.fiber; fiber?.runtime; fiber = fiber.parent.fiber) depth++;
  return depth;
}

export function installPresetGate(ctx, config = {}, { legacy = false } = {}) {
  const depth = entryDepth(ctx);
  const previous = installedGates.get(ctx.root);
  if (previous) {
    if (legacy && !previous.legacy) return;
    if (legacy === previous.legacy && (depth > previous.depth || (depth === previous.depth && previous.active))) return;
    previous.active = false;
    previous.disposeListener?.();
  }
  const gate = { depth, legacy, active: true };
  installedGates.set(ctx.root, gate);
  ctx.effect(() => () => { gate.active = false; });
  const enabled = () => {
    const value = config.takeoverPresets ?? true;
    return typeof value === "boolean" ? value : value.get();
  };
  const seen = new Set();
  gate.disposeListener = ctx.on("internal/config", function (_raw, next) {
    const resolved = next();
    if (!gate.active || !enabled()) return resolved;
    const loadable = packageLoadable();
    const rewritten = rewritePresetConfig(resolved, loadable, config);
    const id = rewritten?.id || rewritten?.config?.id;
    if (isPreset(resolved) && id && !seen.has(String(id))) {
      seen.add(String(id));
      const engine = engineEntry(rewritten);
      if (engine) {
        const config = engine.config || {};
        ctx.logger?.info?.(`spine: 预设 ${id} 已换包 ${engine.name} thresholdRatio=${config.thresholdRatio} headroomTokens=${config.headroomTokens} retainTokens=${config.retainTokens} compactionRetries=${config.compactionRetries}`);
      } else if (!loadable) {
        ctx.logger?.warn?.(`spine: 本会话未接管预设 ${id}，仍用原版压缩。`);
      }
    }
    return rewritten;
  }, { global: true });
}

// Legacy entry for existing compositions; new installs only register compaction-spine.
export default class LiveCompactGate {
  constructor(ctx, config) {
    installPresetGate(ctx, config, { legacy: true });
  }
}
