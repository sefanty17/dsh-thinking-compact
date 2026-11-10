# dsh-thinking-compact

把 DeepSeek Harness（DSH）历史轮次里的**思考链**压成一段第一人称摘录卡片，工具调用与工具结果原文照抄。

> 状态：插件已成型、离线测试通过，**尚未在真实 profile 里跑通**。见 [还没做完的](#还没做完的)。

---

## 解决什么问题

一段对话跑久了，思考链是上下文里最贵也最没用的部分：它每一步都要被重新读一遍，却很少再被用到。工具输出至少还能当事实依据，思考过程在得出结论之后就只是过程。

本插件在每轮结束时检查，挑一段已经翻过去的完整助手轮，把里面的 reasoning 交给模型缩句——保留第一人称的推理口气、转折、判断依据、数值和函数名，删掉重复验算和填充——然后用一条 user 卡片把整段事件替换掉。

**压的是思考链，不是工具。** 切段里工具调用与结果原文照抄；`tool-call` 没等到对应 `tool-result` 的轮次一律不动，否则会留下孤立 tool_use，上游直接 400。

## 为什么不直接改 `reasoning_content`

带 tools 的请求里 `reasoning_content` 必须原样回传，改写会 400；Anthropic 和 Gemini 的 thinking 签名同理。所以遮盖的对象是**一轮会话事件**，不是某个 API 字段：被切轮次的 assistant 消息替换成一条卡片后，它压根不在发给上游的 messages 里，协议校验无从触发。

## 怎么工作

官方只开放一个扩展点：实现 `CompactionEngine`。本插件继承 `@deepseek-ai/dsh-compaction-basic` 的 `BasicCompactionEngine`，通过 `cordis.patch.yml` disable 官方 `compaction-basic` 并 insert 自己（`ctx.compaction` 是单例，只能有一个引擎）。

落刀长这样：

```js
session.append("user/message", checkpointMessage, {
  surfaceOp: { op: "replace", startSeq: chunk.start, endSeq: chunk.end },
});
await this.ctx.sessions.flush(session);
```

压缩结果是一条 checkpoint 消息，**驻留在会话里**——下一个请求天然带着它，卡片文本不变，上游前缀缓存不用重建。这是插件相对外部代理的根本差别。

### 选段的三条硬规则

1. **只切完整助手轮**：每个 `tool-call` 都必须有对应 `tool-result`。
2. **不跨真人消息、不跨半截轮次**；checkpoint 卡片是地板，切点必须在它之后。
3. **双阈值**：思考 token ≥ `reasoningTokens`（默认 4000），切段 token ≥ `chunkMin`。

摘要不合格（空 / JSON / 不比原文短）、备份失败、span 在提交前被改动 —— 任何一条不满足就整刀放弃，原文完整保留。宁可少压，不压坏。

### 缩句提示词

摘要走**第一人称连续段落纯文本**，不做第三人称纪要、不做 JSON。

这不是偏好，是实测结论：先前用 `事实/反应/决定` 三段式 JSON，模型不信任二手纪要、反复重做已完成的工作，步数从 29 涨到 45（+55%）；换成缩句后两组持平在 68 vs 68。

提示词唯一事实来源是 [`bench/samples/compress_prompt.txt`](bench/samples/compress_prompt.txt)，插件运行时读 [`plugin/prompt.txt`](plugin/prompt.txt)，两者逐字节一致，有测试守着。

## 装法

```sh
git clone https://github.com/sefanty17/dsh-thinking-compact.git
cd dsh-thinking-compact
dsh plugin --profile headless add ./plugin
```

## 目录

```
plugin/
  shell.js          薄入口：加载失败也不能把宿主拖死
  engine.js         引擎：fork 自 dsh-live-compact，改了选段/卡片/摘要三处
  reasoning.js      本项目的实质内容：选段、卡片渲染、提示词装载、摘要校验
  prompt.txt        缩句提示词（与 bench 定版逐字节一致）
  plan.js shadow.js raw.js    沿用上游：工具结果摘录、shadow 备份、原文捕获
  gate.js settings.js adapter.js cooldown.js thunder.js   沿用上游
  test/reasoning.test.mjs     离线测试，不需要 DSH 运行时
bench/
  steps.yaml        30 轮长程任务定义（VoxeLibre 分层实现）
  steps.py score.py 任务读取与验收评分（纯标准库，与实现语言无关）
  samples/          真实思考样本 + 定版提示词
outputs/
  ab-5round-report.md  5 轮 A/B 实测报告（代理时代的数据，结论仍然有效）
```

## 实测

5 轮长程 A/B，逐字相同输入，模型 `deepseek-flash`：

| 指标 | A（原版 DSH） | B（+ 压缩） |
|---|---|---|
| 完成轮次 | 5 / 5 | 5 / 5 |
| 验收通过率 | 1.0 | 1.0 |
| 总步数 | 68 | 68 |
| 总 token | — | −8.3% |
| 成本 | ¥0.631 | ¥0.872（+38%） |

那一轮跑的是**外部代理版**。token 省了但成本反而涨，原因已经查清：代理只改转发给上游的 payload、不回写 DSH 会话，于是每个请求都从原文重压一遍，卡片每轮措辞不同，切点处前缀断裂，缓存未命中的单价是命中价的三十倍，省下的 token 全赔进去。

插件版把这个问题从架构上消掉了（压缩结果驻留会话、卡片固定），但**成本还没重测**。完整归因见 [`outputs/ab-5round-report.md`](outputs/ab-5round-report.md)。

## 还没做完的

- **没在真实 profile 里跑通**。`shell.js` 的加载链路（含 peer 依赖解析）未验证，`compaction/summary` 事件是否真落进会话日志未验证。
- **shadow 备份没接进切段路径**。当前只沿用上游对工具结果的 raw 捕获，被缩句的思考原文还没做独立备份。
- **`plan.js` 里约 1100 行工具结果摘录启发式现在用不到**，删之前要先确认 `shadowSections` 等被复用的导出。
- **压测脚本要重写**。`bench/runner.py` 是围着代理写的，已删；30 轮全量需要新的 runner 走「装插件 → 跑 DSH → 读会话日志」。
- 摘要质量只做过端到端任务成功率评测，没单独评摘要文本——摘要本身是另一次生成，单独评它没有证据效力。
- 5 轮样本量偏小。

## 致谢

骨架 fork 自 [MerkyorLynn/dsh-live-compact](https://github.com/MerkyorLynn/dsh-live-compact)（MIT，见 [`plugin/LICENSE.upstream`](plugin/LICENSE.upstream)）：切段、shadow 备份、`surfaceOp: replace` 卡片遮盖、commit 前的一整套校验都是它的。本项目把同一套机制从**工具结果**迁移到**思考链**，差别在于思考链走模型缩句而不是规则截断，并且默认不碰工具。

设计过程中参考过 [T-Auto/dsh-ops](https://github.com/T-Auto/dsh-ops) 的工具链。

## License

[MIT](LICENSE) © 2026 sefanty17。`plugin/` 下 fork 自上游的部分版权归其作者，见 `plugin/LICENSE.upstream`。
