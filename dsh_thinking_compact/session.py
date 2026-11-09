"""按会话驻留的"发送视图"。

## 为什么需要它

DSH 每次请求都把完整原文发过来。代理如果每次都从原文重压一遍，会连撞三件事：

1. 每个请求都付一次摘要调用的延迟（长程任务里能占掉一半墙钟）；
2. 卡片措辞每次略有不同，切点之后的上游前缀立刻失效；
3. 缓存未命中的单价是命中价的三十倍，省下来的 token 全赔进去。

## 做法

给每个会话维护一份"实际发给上游的视图"，**只追加、不回改**：

- 新请求先和已吸收的原文逐条比指纹，对不上就截断到公共前缀重来；
- 只有"还没发出去过"的增量允许压缩 —— 压缩永远发生在尾部，
  已经发出去的字节一个都不动，前缀缓存因此可以一路续下去；
- 卡片生成一次就固化在视图里，后续请求原样复用，不再重算。

视图里每一项都记着自己覆盖的原文下标区间（卡片是闭区间，普通消息是单点），
所以截断、比对、统计都能回到原文坐标。

## 边界

- 增量里切不动（阈值不够、摘要不合格、备份失败）就原样吸收，绝不半途改；
- 切点之后若紧跟真人消息，会出现两条连续 user 消息，这是整段替换的固有形态
  （参照物同样如此），严格交替的 provider 需要 `merge_card_into_next_user`。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .card import build_card
from .config import Config
from .messages import (
    MsgSpan,
    _backup_view,
    message_text,
    select_tail,
)
from .shadow import ShadowError, log_cut, save_shadow
from .summarizer import summarize

KIND_MSG = "msg"
KIND_CARD = "card"


def message_digest(message: dict) -> str:
    """一条消息的稳定指纹。用于判定"这条原文我是不是已经吸收过"。"""
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def fingerprint_of(system, messages: list[dict]) -> str:
    """会话指纹：system + 第一条真人消息。

    只取开头是为了让同一会话里后续所有请求都落到同一个状态文件上，
    同时不同会话（哪怕提示词很像）不会互相串。
    """
    h = hashlib.sha256()
    if system is not None:
        h.update(json.dumps(system, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for message in messages:
        if message.get("role") == "user":
            h.update(json.dumps(message, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            break
    return h.hexdigest()[:16]


def _tokens(text: str) -> int:
    return max(1, len(text) // 3) if text else 0


@dataclass
class Item:
    """发送视图里的一项：要么是一条原样消息，要么是一张盖住若干条原文的卡片。"""
    kind: str
    start: int          # 覆盖的原文下标（闭区间）
    end: int
    message: dict

    def to_dict(self) -> dict:
        return {"kind": self.kind, "start": self.start, "end": self.end, "message": self.message}

    @classmethod
    def from_dict(cls, raw: dict) -> "Item":
        return cls(
            kind=str(raw.get("kind") or KIND_MSG),
            start=int(raw["start"]),
            end=int(raw["end"]),
            message=raw["message"],
        )


@dataclass
class SessionState:
    fingerprint: str
    hashes: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    @property
    def absorbed(self) -> int:
        """已经吸收到第几条原文（= 发送视图覆盖的原文长度）。"""
        return self.items[-1].end + 1 if self.items else 0

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "hashes": self.hashes,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SessionState":
        return cls(
            fingerprint=str(raw["fingerprint"]),
            hashes=[str(h) for h in raw.get("hashes") or []],
            items=[Item.from_dict(i) for i in raw.get("items") or []],
        )


class ResidentSession:
    """一个会话的驻留视图。所有读写都在锁里，避免并发请求互相踩。"""

    def __init__(self, state: SessionState, cfg: Config, path: Path | None = None):
        self.state = state
        self.cfg = cfg
        self.path = path
        self.lock = threading.Lock()

    # ---- 持久化 -------------------------------------------------------

    def _save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.state.to_dict(), fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError:
            # 落盘失败不影响这一轮：视图还在内存里，下个请求照样能续
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    # ---- 前缀校验 -----------------------------------------------------

    def _truncate(self, keep: int) -> None:
        """把视图截回只覆盖原文前 keep 条。卡片若跨过 keep 就整张丢掉。"""
        kept: list[Item] = []
        for item in self.state.items:
            if item.end < keep:
                kept.append(item)
            else:
                break
        self.state.items = kept
        absorbed = kept[-1].end + 1 if kept else 0
        self.state.hashes = self.state.hashes[:absorbed]

    def _resync(self, messages: list[dict]) -> int:
        """返回已吸收的长度；原文前缀对不上时先把视图截断。"""
        limit = min(len(self.state.hashes), len(messages))
        keep = 0
        while keep < limit and self.state.hashes[keep] == message_digest(messages[keep]):
            keep += 1
        if keep < self.state.absorbed:
            self._truncate(keep)
        return self.state.absorbed

    # ---- 吸收 ---------------------------------------------------------

    def _absorb_plain(self, chunk: list[dict], offset: int) -> None:
        for k, message in enumerate(chunk):
            self.state.items.append(Item(KIND_MSG, offset + k, offset + k, message))
            self.state.hashes.append(message_digest(message))

    def _absorb_card(self, card: str, start: int, end: int) -> None:
        self.state.items.append(Item(
            KIND_CARD, start, end, {"role": "user", "content": [{"type": "text", "text": card}]}
        ))

    # ---- 主流程 -------------------------------------------------------

    def _try_cut(self, tail: list[dict], offset: int, llm_call):
        """在增量里切一刀。切不动返回 (None, 原因)。"""
        span, reason = select_tail(tail, self.cfg)
        if span is None:
            return None, reason
        if llm_call is None:
            return None, "no_llm_call"

        segment = tail[span.start:span.end + 1]
        global_span = MsgSpan(
            offset + span.start,
            offset + span.end,
            span.reasoning_tokens,
            span.segment_tokens,
            span.thinking_text,
        )
        try:
            backup = save_shadow(_backup_view(segment), global_span, self.cfg)
        except ShadowError:
            return None, "backup_failed"

        summary = summarize(span.thinking_text, llm_call, ratio_max=self.cfg.target_ratio_max)
        if summary is None:
            return None, "summary_invalid"

        card = build_card(
            span=global_span,
            segment=segment,
            summary=summary,
            backup_id=backup.backup_id,
            digest=backup.digest,
            nbytes=backup.nbytes,
        )
        return (span, global_span, backup, card), "ok"

    def render(self, messages: list[dict], llm_call=None) -> tuple[list[dict], dict]:
        """算出这一轮实际发给上游的 messages，并把视图推进到最新原文。"""
        with self.lock:
            before = sum(_tokens(message_text(m)) for m in messages)
            absorbed = self._resync(messages)
            tail = messages[absorbed:]

            stats = {
                "resident": True,
                "compacted": False,
                "reason": "nothing_new" if not tail else "not_cut",
                "absorbed_before": absorbed,
                "tail": len(tail),
                "cards": sum(1 for i in self.state.items if i.kind == KIND_CARD),
            }
            if not tail:
                stats["tokens_before"] = before
                stats["tokens_after"] = sum(
                    _tokens(message_text(i.message)) for i in self.state.items
                )
                return [item.message for item in self.state.items], stats

            result, reason = self._try_cut(tail, absorbed, llm_call)

            if result is None:
                stats["reason"] = reason
                self._absorb_plain(tail, absorbed)
            else:
                span, global_span, backup, card = result
                local_start, local_end = span.start, span.end
                self._absorb_plain(tail[:local_start], absorbed)
                self._absorb_card(card, absorbed + local_start, absorbed + local_end)
                # 卡片覆盖的原文也要进指纹表，下一轮才能正确比对前缀
                for k in range(local_start, local_end + 1):
                    self.state.hashes.append(message_digest(tail[k]))
                self._absorb_plain(tail[local_end + 1:], absorbed + local_end + 1)

                card_tokens = _tokens(card)
                log_cut(global_span, backup, card_tokens, self.cfg)
                stats.update({
                    "compacted": True,
                    "reason": "ok",
                    "span": [global_span.start, global_span.end],
                    "rounds": span.rounds,
                    "reasoning_tokens": span.reasoning_tokens,
                    "card_tokens": card_tokens,
                    "backup": backup.backup_id,
                })

            view = [item.message for item in self.state.items]
            stats["absorbed_after"] = self.state.absorbed
            stats["cards"] = sum(1 for i in self.state.items if i.kind == KIND_CARD)
            stats["tokens_before"] = before
            stats["tokens_after"] = sum(_tokens(message_text(m)) for m in view)
            self._save()
            return view, stats


class SessionStore:
    """按会话指纹分发 ResidentSession，状态落在 sessions/<fp>.json。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.sessions_dir
        self._sessions: dict[str, ResidentSession] = {}
        self._guard = threading.Lock()

    def _path(self, fingerprint: str) -> Path:
        return self.root / (fingerprint + ".json")

    def _load(self, fingerprint: str) -> SessionState:
        path = self._path(fingerprint)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                state = SessionState.from_dict(json.load(fh))
            if state.fingerprint == fingerprint:
                return state
        except (OSError, ValueError, KeyError):
            pass
        return SessionState(fingerprint=fingerprint)

    def get(self, fingerprint: str) -> ResidentSession:
        with self._guard:
            session = self._sessions.get(fingerprint)
            if session is None:
                session = ResidentSession(self._load(fingerprint), self.cfg, self._path(fingerprint))
                self._sessions[fingerprint] = session
            return session

    def render(self, payload: dict, llm_call=None) -> tuple[list[dict], dict]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return messages, {"resident": True, "compacted": False, "reason": "no_messages"}
        fingerprint = fingerprint_of(payload.get("system"), messages)
        session = self.get(fingerprint)
        view, stats = session.render(messages, llm_call)
        stats["fingerprint"] = fingerprint
        return view, stats
