/**
 * 加载链路探针。放在 profile 的 node_modules/dsh-thinking-compact 里跑，
 * 这样 `@deepseek-ai/*` 能沿父目录解析。不需要 API key，也不挂宿主。
 */
const shell = await import("./shell.js");
const engine = await import("./engine.js");
const reasoning = await import("./reasoning.js");

const entry = shell.default;
console.log("shell 默认导出:", typeof entry, entry?.name);
console.log("静态 inject:", JSON.stringify(entry?.inject));
console.log("Config schema:", entry?.Config ? "present" : "missing");
console.log("engine 默认导出:", typeof engine.default, engine.default?.name);
console.log("engine 基类:", Object.getPrototypeOf(engine.default)?.name);
console.log("engine 有 summarize:", typeof engine.default?.prototype?.summarize);
console.log("engine 有 commitSelectedSpan:", typeof engine.default?.prototype?.commitSelectedSpan);
console.log("提示词字符数:", reasoning.PROMPT.length);
console.log("占位符存在:", reasoning.PROMPT.includes(reasoning.PLACEHOLDER));
