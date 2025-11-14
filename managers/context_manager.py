from __future__ import annotations

from typing import Dict, List

import numpy as np

from managers.embedding import EmbeddingManager


class ContextManager:
    MAX_HISTORY = 10  # 최근 10개 대화만 유지
    SUMMARIZE_AFTER = 20  # 20개 이상이면 요약 실행

    def __init__(self, llm=None, embedder: EmbeddingManager | None = None):
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.session_summary: dict[str, str] = {}
        self.llm = llm  # 요약용 LLMManager 인스턴스
        self.embedder = embedder
        self._context_embedding_cache: Dict[str, np.ndarray | None] = {}

    def add_message(self, tab_id: str | int | None, role: str, content: str):
        key = self._context_key(tab_id)
        self.sessions.setdefault(key, []).append({"role": role, "content": content})

        # 일정 개수 초과 시 요약
        if len(self.sessions[key]) > self.SUMMARIZE_AFTER:
            self._summarize_and_trim(key)
        self._invalidate_embedding(key)

    def _summarize_and_trim(self, key: str):
        """이전 대화를 요약하고, 최신 n개만 남김"""
        messages = self.sessions[key]
        old = messages[:-self.MAX_HISTORY]
        recent = messages[-self.MAX_HISTORY:]
        if not self.llm or not old:
            self.sessions[key] = recent
            return

        text_to_summarize = "\n".join([f"{m['role']}: {m['content']}" for m in old])
        summary = self.llm.generate(
            f"Summarize this conversation briefly (in Korean):\n{text_to_summarize}",
            task="summarization"
        )
        prev_summary = self.session_summary.get(key, "")
        self.session_summary[key] = prev_summary + "\n" + summary
        self.sessions[key] = recent
        self._invalidate_embedding(key)

    def build_prompt(self, tab_id: str | None, user_text: str, *, include_history: bool = True) -> str:
        key = self._context_key(tab_id)
        if include_history:
            context_block = self._compose_context_text(key)
        else:
            context_block = (
                "### Conversation Summary ###\n(History omitted due to low similarity)\n"
                "### Recent Messages ###\n(omitted)"
            )

        lines = [context_block, "### End ###", f"User: {user_text}"]
        lines.append(
            "\nIf user asks about earlier messages, refer to the numbered list or the summary above."
        )
        return "\n".join(lines)

    def export_context_text(self, tab_id: str | int | None) -> str:
        key = self._context_key(tab_id)
        return self._compose_context_text(key)

    def get_context_embedding(self, tab_id: str | int | None) -> np.ndarray | None:
        if not self.embedder:
            return None
        key = self._context_key(tab_id)
        cached = self._context_embedding_cache.get(key)
        if cached is not None:
            return cached
        context_text = self._compose_context_text(key).strip()
        if not context_text:
            return None
        embedding = self.embedder.embed_text(context_text, command="document")
        self._context_embedding_cache[key] = embedding
        return embedding

    def context_similarity(self, tab_id: str | int | None, user_text: str) -> float | None:
        if not self.embedder or not user_text or not user_text.strip():
            return None
        context_emb = self.get_context_embedding(tab_id)
        if context_emb is None or not context_emb.any():
            return None
        user_emb = self.embedder.embed_text(user_text, command="query")
        if not user_emb.any():
            return 0.0
        score = float(np.dot(user_emb, context_emb))
        return max(-1.0, min(1.0, score))

    def _compose_context_text(self, key: str) -> str:
        context = self.sessions.get(key, [])
        summary = self.session_summary.get(key, "(No previous summary)")
        lines = ["### Conversation Summary ###", summary, "### Recent Messages ###"]
        for i, msg in enumerate(context[-self.MAX_HISTORY:], start=1):
            role = msg.get("role", "").capitalize() or "User"
            lines.append(f"[{i}] {role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _invalidate_embedding(self, key: str):
        self._context_embedding_cache.pop(key, None)

    @staticmethod
    def _context_key(tab_id: str | int | None) -> str:
        return str(tab_id or "__default__")
