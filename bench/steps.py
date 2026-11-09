"""steps.yaml 的最小读取器。

bench 不允许引入第三方依赖，而 steps.yaml 只用到 YAML 的一个很窄的子集：
顶层列表 + 缩进 2 的标量键 + `|` 块标量 + 缩进 4 的 acceptance 列表。
这里按这个子集解析，遇到不认识的构造直接报错，不猜。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Step:
    round: int
    title: str
    goal: str
    spec: str
    acceptance: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    reference: str = ""


def _scalar(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] == "'":
        # YAML 单引号：反斜杠是字面量，只有 '' 表示一个单引号
        return text[1:-1].replace("''", "'")
    if len(text) >= 2 and text[0] == text[-1] and text[0] == '"':
        return ast.literal_eval(text)
    return text


def _block(lines: list[str], start: int, indent: int) -> tuple[str, int]:
    """从 start 开始收集缩进 >= indent 的行，去掉 indent 前缀。"""
    out: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            out.append("")
            i += 1
            continue
        leading = len(line) - len(line.lstrip(" "))
        if leading < indent:
            break
        out.append(line[indent:])
        i += 1
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out), i


def load_steps(path: str | Path) -> list[Step]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    steps: list[Step] = []
    cur: dict | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue

        if line.startswith("- round:"):
            if cur is not None:
                steps.append(Step(**cur))
            cur = {"round": int(line.split(":", 1)[1].strip()), "title": "", "goal": "",
                   "spec": "", "acceptance": [], "depends_on": [], "reference": ""}
            i += 1
            continue
        if cur is None:
            raise ValueError("steps.yaml: 数据出现在第一条 - round 之前: %r" % line)

        if line.startswith("    - "):
            # acceptance 条目
            item = line[6:].strip()
            if item == "|":
                text, i = _block(lines, i + 1, 6)
                cur["acceptance"].append(text)
            else:
                cur["acceptance"].append(_scalar(item))
                i += 1
            continue

        if line.startswith("  ") and not line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip()
            if value == "|":
                text, i = _block(lines, i + 1, 4)
                cur[key] = text
                continue
            if key == "acceptance":
                cur["acceptance"] = []
                i += 1
                continue
            if key == "depends_on":
                inner = value.strip()
                if not (inner.startswith("[") and inner.endswith("]")):
                    raise ValueError("steps.yaml: depends_on 必须是内联列表: %r" % value)
                cur[key] = [int(x) for x in inner[1:-1].split(",") if x.strip()]
                i += 1
                continue
            cur[key] = _scalar(value)
            i += 1
            continue

        raise ValueError("steps.yaml: 无法解析第 %d 行: %r" % (i + 1, line))

    if cur is not None:
        steps.append(Step(**cur))
    return steps


def prompt_for(step: Step) -> str:
    """按 README 的投喂规则拼这一轮的用户消息。"""
    acceptance = "\n".join(step.acceptance)
    return (
        "第 %d 轮：%s\n\n"
        "目标：\n%s\n\n"
        "规格：\n%s\n\n"
        "验收命令（每条都必须退出码 0）：\n%s\n\n"
        "完成后运行本轮 acceptance 的全部命令，并把输出贴回来。"
        % (step.round, step.title, step.goal, step.spec, acceptance)
    )
