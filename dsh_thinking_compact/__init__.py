"""dsh_thinking_compact：把 agent 历史里过长的思考链压成固定格式卡片。

对外只暴露一个主入口 compact_messages，其余模块可按需单独引用：
    config      阈值与路径
    selector    选"刚结束的一轮"
    summarizer  三段摘要 + 校验
    card        卡片文本
    shadow      原文备份 / 逐刀日志
    apply       主流程
    proxy       本地 HTTP 代理（python -m dsh_thinking_compact.proxy）
    demo        离线演示（python -m dsh_thinking_compact.demo）
"""
from .apply import CompactResult, Stats, compact_messages, stats_dict
from .config import Config

__all__ = ["Config", "CompactResult", "Stats", "compact_messages", "stats_dict"]
