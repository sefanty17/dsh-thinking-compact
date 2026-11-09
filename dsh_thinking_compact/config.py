"""配置：阈值、备份目录、压缩比上限。

所有阈值都能被显式传参覆盖；运行期只开放 THINK_MIN_TOKENS 一个环境变量，
其余走构造参数，避免环境变量散落成隐式行为。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 备份根目录：~/.dsh/thinking-compact/shadow
# 用 default_factory 而不是类属性，保证迁移 HOME 之后仍然指向当前用户目录。
def default_shadow_dir() -> Path:
    return Path.home() / ".dsh" / "thinking-compact" / "shadow"


@dataclass(frozen=True)
class Config:
    # 该轮 assistant 的 reasoning token 之和达到这个值才考虑切
    min_reasoning_tokens: int = 4000
    # 切段自身 token 不足这个值就不切，避免为一点内容付一次模型调用
    chunk_min_tokens: int = 1000
    # 切段内最近的 N 个工具轮保持原文（0 = 不保留，所有可切轮次都压，正式 A/B 用）
    keep_rounds: int = 0
    # 整段历史超过这个长度就不做小切，交给上游自己的大压缩
    max_context_tokens: int = 650_000
    # 压缩比上限：原文 / 摘要 字符数超过它，视为有遗漏风险，判定不合格
    target_ratio_max: float = 8.0
    # 原文备份目录
    shadow_dir: Path = field(default_factory=default_shadow_dir)
    # 代理转发与摘要调用的超时（秒）；缩句输出较长，摘要调用需要更宽的超时
    max_upstream_seconds: float = 300.0
    # 按会话驻留：只压新追加的增量、卡片生成一次就固化。
    # 关掉退回旧行为（每个请求都从完整原文重压一遍），只用于对照实验。
    resident: bool = True

    @property
    def usage_path(self) -> Path:
        """usage.jsonl 与 shadow 同父目录：~/.dsh/thinking-compact/usage.jsonl。"""
        return self.shadow_dir.parent / "usage.jsonl"

    @property
    def sessions_dir(self) -> Path:
        """会话驻留视图：~/.dsh/thinking-compact/sessions/<fingerprint>.json。"""
        return self.shadow_dir.parent / "sessions"

    @property
    def cuts_path(self) -> Path:
        """逐刀明细日志，只记序号/token/备份 id，不记原文。"""
        return self.shadow_dir / "spine-cuts.jsonl"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """读 THINK_MIN_TOKENS，其余用参数覆盖。"""
        env: dict = {}
        raw = os.environ.get("THINK_MIN_TOKENS")
        if raw:
            try:
                env["min_reasoning_tokens"] = int(raw)
            except ValueError:
                # 环境变量写错不静默改阈值，直接用默认值
                pass
        env.update(overrides)
        return cls(**env)
