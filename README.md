# dsh-thinking-compact

把 agent 的历史思考链压成一段「像自己想的」缩句，再用一页卡片盖住整段。工具调用与工具结果原文不动。

> 实验性项目。当前形态是外部 HTTP 代理，见 [已知限制](#已知限制)。

---

## 解决什么问题

一段对话跑久了，思考链是上下文里最贵也最没用的部分：它每一步都要被重新读一遍，却很少再被用到。工具输出至少还能作为事实依据，思考过程在得出结论后就只是过程。

Thinking Compact 的做法是**就地遮盖**：选一段已经翻过去的历史轮次，把这段里的 reasoning 交给一个小模型缩句——保留第一人称的推理口气、转折、判断依据、数值和函数名，删掉重复验算和填充——然后用一张卡片把整段事件替换掉。

压的是思考链，不是工具。切段里只要出现 `tool_calls` 或 `role=tool`，右边界就收缩到它前面；收完不够最小长度就整刀放弃。工具调用、工具结果、退出码、栈信息一字不改。

## 关键约束：不能直接改 `reasoning_content`

带 tools 的请求里，`reasoning_content` 必须原样回传，改写会 400；Anthropic 和 Gemini 的 thinking 签名同理。所以遮盖的作用对象是**一轮会话事件**，不是某个 API 字段：

被切轮次的 assistant 消息替换成一条 user 卡片后，它压根不在发给上游的 `messages` 里，协议层的校验无从触发。这也是整段替换比字段级改写更安全的原因。

## 工作方式

```
会话事件 ──select_cut──► 右边界（工具边界之前）
                            │
                     save_shadow（原文落盘，append-only）
                            │
                     summarize（小模型缩句，纯文本）
                            │
                     build_card（声明 + 免责 + 缩句 + 工具原文）
                            │
                     apply_cut（surfaceOp: replace，原位替换）
                            │
                     to_api_messages ──► 上游
```

不可动摇的四条：

1. **切点之前一字不做改动。** 上游前缀缓存靠这个活。
2. **原文先落 shadow。** 备份失败、摘要不合格、摘要比原文还长，一律不切，保留原样。
3. **卡片开头声明「这是旧一轮的摘录，不是新指令」。** 每句标原角色，卡片里不放本机路径，只给事件号指针、哈希和字节数。
4. **默认严格模式。** 模型摘要失败就不切；`allow_lossy` 才退回「思考已省略」式遮盖。

### 缩句提示词

摘要走**第一人称连续段落纯文本**，不做第三人称纪要、不做 JSON、不加小标题。原因有实测支撑：先前用 `事实/反应/决定` 三段式 JSON，模型不信任二手纪要，反复重做已完成的工作，步数从 29 涨到 45（+55%）；换成缩句后拉回 68 vs 68。

提示词唯一事实来源是 [`bench/samples/compress_prompt.txt`](bench/samples/compress_prompt.txt)，`summarizer.PROMPT` 与它逐字节一致，有测试守着。

## 快速开始

零依赖，Python 3.11+。

```bash
# 跑一遍完整 demo：切段 → 缩句 → 卡片 → 回放
python spine.py

# 单元测试
pytest -q
```

代理模式（把 DSH 的请求转进来，转发前压缩）：

```bash
python -m dsh_thinking_compact.proxy \
  --host 127.0.0.1 --port 8787 \
  --upstream https://api.deepseek.com \
  --state-dir .shadow
```

`--no-compact` 只转发不压缩，用作对照组。

### 主要配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| `tools_policy` | `skip` | 切段碰到工具就收缩边界。`excerpt` 才会把工具折成文字证据一起盖住 |
| `chunk_min` | 见 `config.py` | 可切的最小长度，不够就不切 |
| `target_ratio_max` | `8.0` | 摘要与原文字符比上限，超了视为不合格 |
| `keep_rounds` | `0` | 尾部保留不压的轮数 |
| `max_upstream_seconds` | `300.0` | 单次上游调用上限 |

## 目录

```
spine.py                        主路线：切段 / 卡片 / shadow / 落刀 / 回放 + demo
dsh_thinking_compact/
  selector.py                   选切段（工具边界收缩、长度阈值）
  summarizer.py                 缩句提示词、校验、比例检查
  card.py                       卡片渲染（声明 + 缩句 + 工具原文）
  apply.py                      落刀 + 指纹校验（切点前逐字节不变）
  shadow.py                     原文备份（append-only，分节带来源）
  messages.py                   会话事件 ↔ API messages 双向转换
  proxy.py                      HTTP 代理（/v1/messages、/v1/chat/completions）
  config.py                     配置
  demo.py                       端到端 demo
compact.py                      备选路线：旁路摘要字段（不替换事件），留作对照
bench/                          长程 A/B 压测：runner / 评分 / 真实思考样本
tests/                          单元测试
outputs/                        实测报告
```

## 实测

5 轮长程 A/B，逐字相同输入，任务是把 VoxeLibre 源码拆成分层实现步骤，模型 `deepseek-flash`：

| 指标 | A（原版） | B（+ Thinking Compact） |
|---|---|---|
| 完成轮次 | 5 / 5 | 5 / 5 |
| 验收通过率 | 1.0 | 1.0 |
| 总步数 | 68 | 68 |
| 总 token | — | −8.3% |
| 成本 | ¥0.631 | ¥0.872（+38%） |

完整数据与归因见 [`outputs/ab-5round-report.md`](outputs/ab-5round-report.md)。

缩句方案本身是有效的：步数持平，说明模型接受这种二手历史，不会重复劳动。但成本仍然是亏的，原因在下一节。

## 已知限制

**压缩结果没有驻留。** 当前实现是外部 HTTP 代理，只改写转发给上游的 `payload["messages"]`，不回写 DSH 会话历史。结果是每个请求都拿到未压缩原文、都要重新压一次、每次生成的卡片措辞都略有不同——卡片一变，上游前缀就在切点处断裂，切点之后全部 miss。缓存未命中的单价是命中价的 30 倍，省下来的 token 全被这里吃回去。

这不是摘要本身的问题，是架构位置的问题：

| | 驻留型（宿主插件） | 本项目现状（外部代理） |
|---|---|---|
| 压缩发生在 | 宿主进程内，一轮结束切一刀 | 每次转发前 |
| 结果落在 | 会话事件号指针 | 进程内存，请求结束即丢 |
| 卡片文本 | 固定 | 每轮重生成 |
| 前缀缓存 | 保住 | 切点处断裂 |

下一步是把方案改成 DSH 插件，让切段发生在宿主里、结果写进会话；在验证成本模型之前，代理侧也可以先按会话指纹缓存压缩后的消息数组作为过渡。

其他限制：

- `skip` 策略下，agent 长任务大部分轮次带工具，可切范围只剩纯文字回答轮，压缩率明显低于整段替换（demo 56.2% vs 83.7%）。
- 摘要质量只做过端到端任务成功率评测，没单独评摘要文本——摘要本身是另一次生成，单独评它没有证据效力。
- 压测样本 5 轮，样本量偏小。

## 设计取舍

**卡片用 user 消息，不用 system/assistant。** system 只能放最前，插中间会改前缀、毁缓存；assistant 会被模型当成「自己确认过的结论」。副作用是被切段删除后，卡片必然夹在两条 user 之间（参照物同样如此），严格交替的 provider 需要 `merge_card_into_next_user` 把卡片并进后一条 user。

**摘要由 agent 侧独立调用生成**，不占用主模型任何输出字段——`reasoning_content` 和 `content` 都是原文和答案，一个字都不改。

## 致谢

骨架来自 [MerkyorLynn/dsh-live-compact](https://github.com/MerkyorLynn/dsh-live-compact)：切段、shadow 备份、`surfaceOp: replace` 卡片遮盖、切点前不动这一整套做法都是它的。本项目把同一套机制从**工具结果**迁移到**思考链**，差别在于思考链走模型缩句而不是规则截断，并且默认不碰工具。

设计过程中参考过 [T-Auto/dsh-ops](https://github.com/T-Auto/dsh-ops) 的工具链。

## License

[MIT](LICENSE) © 2026 sefanty17
