# bench / steps.yaml 使用说明

这是一份用于 A/B 长程任务测试的 30 轮脚本。目标不是复刻 VoxeLibre，而是让被测 agent 在 30 次对话里，从零搭出一个可运行的“方块注册 + 世界生成 + 放置/破坏 + 存档”最小核心。步骤只描述“要实现什么、怎么验收”，不包含 VoxeLibre 源码。

## 目标技术栈与目录约定

- Python 3.10+，仅标准库；不访问网络。
- 所有命令默认在**项目根目录**执行，根目录下必须有 `game/` 包。
- 建议产物：`game/blocks.py`、`builtins.py`、`chunk.py`、`world.py`、`noise.py`、`mapgen.py`、`player.py`、`storage.py`、`cli.py`、`demo.py`，以及 `tests/`。
- 方块名用字符串保存（如 `stone`、`coal_ore`），与世界存档格式一致。

## 30 轮怎么发

1. 每轮单独发一条用户消息，只给该轮的 `title`、`goal`、`spec`、`acceptance`，不要提前泄露后续轮次。
2. 消息末尾固定加一句：“完成后运行本轮 acceptance 的全部命令，并把输出贴回来。”
3. 被测 agent 在同一工作目录连续工作；上一轮产物就是下一轮起点。
4. 每轮结束后，由评测脚本在项目根目录重新执行该轮 acceptance，而不是相信 agent 的自述。

## 单轮判定

- 一个 `acceptance` 条目就是一条 shell 命令（`python -c ...` 或 `python - <<PY ... PY`）。
- 命令退出码为 0 记为通过；非 0 或超时（建议单条 120s）记为失败。
- 本轮通过率 `r_i = 通过条数 / 本轮 acceptance 总条数`，取值 0..1。
- 多行 heredoc 条目请用 YAML 解析后 `textwrap.dedent` 再交给 shell，避免缩进导致 `IndentationError`。
- 只检查行为，不检查代码风格；但验收点已覆盖数据结构、确定性、边界与错误路径。

## 总分汇总（按轮加权）

```
total = sum(w_i * r_i) / sum(w_i)      # i = 1..30
```

默认 `w_i = 1`。若想强调长程能力，可用分段权重：第 1-10 轮 1.0，第 11-19 轮 1.2，第 20-30 轮 1.5。另可记录“连续通过轮数”作为辅助指标，避免只看总分掩盖中途崩盘。

## 覆盖范围与参考来源

- 方块注册：`mods/ITEMS/mcl_core/nodes_base.lua`、`nodes_liquid.lua`、`nodes_trees.lua`、`mods/CORE/mcl_autogroup/`。
- 世界生成：`mods/MAPGEN/mcl_mapgen_core/init.lua`、`api.lua`、`ores.lua`，`mods/MAPGEN/mcl_biomes/init.lua`，`mods/ITEMS/mcl_core/schematics/`。
- 玩家交互：`mods/PLAYER/mcl_player/init.lua`、`mods/PLAYER/playerphysics/init.lua`、`mods/ITEMS/mcl_core/functions.lua`。
- 世界元数据与存档：`mods/CORE/mcl_worlds/init.lua`；VoxeLibre 的落盘由 Luanti 引擎负责，故存档轮次只参考其世界元数据组织方式。
- 回归测试组织：`mods/CORE/vl_unittests/`。

## 结果记录建议

每轮记录：`round`、`passed`、`total`、`r_i`、失败命令原文与 stderr 前 20 行、耗时。30 轮跑完后同时给出加权总分、首次失败轮次、以及“回退重试后是否恢复”，用于区分稳定性与能力上限。
