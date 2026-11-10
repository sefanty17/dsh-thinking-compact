# dsh-thinking-compact（插件版）

把 DSH 历史轮次里的**思考链**压成一段第一人称摘录卡片，工具调用与工具结果原文照抄。
宿主是 DSH 本体，压缩结果通过 `surfaceOp: { op: "replace" }` 写进会话事件，天然驻留。

> 状态：选段与卡片已替换，离线测试 11/11；装配链路已在 headless profile 验证；**尚未在真实会话里跑出过一次切刀**。

## 这份代码从哪来

骨架 fork 自 [dsh-live-compact](https://github.com/MerkyorLynn/dsh-live-compact) v1.1.3，已对齐 **v1.1.4**（MIT，
见 `LICENSE.upstream`）。v1.1.4 只有一处改动：把每轮结束时的 raw 补扫从关键路径上摘成
分片后台任务，修掉长会话冻结（[issue #34](https://github.com/Tranquillitatis-Lab/dsh-live-compact/issues/34)）。它解决的是「工具结果太长」，本插件解决「思考链太长」，
所以引擎骨架、checkpoint 协议、shadow 备份、tool pairing 校验全部沿用，只换三处。

## 只换三个函数

| 位置 | 参照物 | 这里 |
|---|---|---|
| `selectNewestChunk`（engine.js，模块级） | 按工具结果的行做摘录启发式 | `reasoning.js#selectReasoningChunk`：只切完整助手轮，思考 token 过阈值 |
| `chunkPlan` / 卡片 | 工具原文 + 事实行 | `reasoning.js#renderCardText`：声明 + 缩句 + 工具原文 + 原文指针 |
| `summarize()` | 通用摘要，`SPINE_THUNDER=1` 才调模型 | `summarizeReasoning`：固定缩句提示词，无条件走模型 |

**一行都没动的**：`commitSelectedSpan` 的全部校验（checkpoint 地板、用户边界、
tool pairing、摘要必须比原文小）、`compaction/start|summary|end` 三事件、
`shadow` 备份与 ACL、`gate` / `settings` / `shell` 入口。

## 选段规则

三条硬约束：

1. **只切完整助手轮**：每个 `tool-call` 必须等到对应 `tool/result`，否则这轮不动
   （切了会留下孤立 tool_use，上游直接 400）。
2. **不跨真人消息、不跨半截轮次**；checkpoint 卡片是地板，切点必须在它之后。
3. **双阈值**：思考 token ≥ `reasoningTokens`（默认 4000），切段 token ≥ `chunkMin`。

摘要不合格（空 / JSON / 不比原文短）直接抛错，`commitSelectedSpan` 会放弃这一刀，
原文完整保留 —— 宁可少压，不压坏。

## 装法

```sh
# 在仓库根目录
dsh plugin --profile headless add ./plugin
```

`cordis.patch.yml` 会 disable 官方 `compaction-basic` 并 insert 本引擎
（`ctx.compaction` 是单例，只能有一个引擎）。

## 测试

```sh
node --test plugin/test/reasoning.test.mjs
```

离线、不依赖 DSH 运行时，用假 session 驱动选段与卡片。

## 还没做的

- 没在真实会话里跑出过一次切刀：`compaction/summary` 是否真落进会话日志、卡片是否出现在下一轮请求里，都未验证。
- `plan.js` 里工具结果摘录那套启发式（约 1100 行）现在用不到了，还没删。
- shadow 备份尚未接入我们的切段路径（当前沿用参照物的 raw 捕获，只覆盖工具结果；被缩句的思考原文没有独立备份）。

## License

MIT。上游部分版权归 dsh-live-compact 作者，见 `LICENSE.upstream`。
