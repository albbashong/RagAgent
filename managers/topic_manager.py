from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TopicThread:
    topic_id: str
    title: str
    embedding: np.ndarray | None
    created_at: datetime
    last_similarity: float = 0.0
    message_count: int = 0


class TopicManager:
    def __init__(self, *, embedder, similarity_threshold: float = 0.6):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self._topics: Dict[str, List[TopicThread]] = {}
        self._active_topic: Dict[str, str] = {}
        self._counter = 0

    def assign_topic(self, tab_id: int | str | None, user_text: str) -> Dict[str, Any]:
        tab_key = self._tab_key(tab_id)
        topics = self._topics.setdefault(tab_key, [])
        user_emb = self._embed(user_text, command="query")
        best_topic: Optional[TopicThread] = None
        best_score = -1.0
        for topic in topics:
            if topic.embedding is None or not topic.embedding.any():
                continue
            score = float(np.dot(user_emb, topic.embedding))
            if score > best_score:
                best_topic = topic
                best_score = score
        if best_topic and best_score >= self.similarity_threshold:
            best_topic.last_similarity = best_score
            best_topic.message_count += 1
            self._active_topic[tab_key] = best_topic.topic_id
            return {
                "topic_id": best_topic.topic_id,
                "similarity": best_score,
                "is_new": False,
                "tab_key": tab_key,
                "title": best_topic.title,
            }
        new_topic = self._create_topic(tab_key, user_text, user_emb)
        new_topic.message_count = 1
        self._active_topic[tab_key] = new_topic.topic_id
        return {
            "topic_id": new_topic.topic_id,
            "similarity": None,
            "is_new": True,
            "tab_key": tab_key,
            "title": new_topic.title,
        }

    def topic_conversation_id(self, tab_id: int | str | None, topic_id: str) -> str:
        return f"{self._tab_key(tab_id)}::topic::{topic_id}"

    def update_topic_embedding(self, tab_id: int | str | None, topic_id: str, embedding: np.ndarray | None):
        if embedding is None:
            return
        topic = self._get_topic(tab_id, topic_id)
        if topic is not None:
            topic.embedding = embedding

    def active_topic_id(self, tab_id: int | str | None) -> Optional[str]:
        return self._active_topic.get(self._tab_key(tab_id))

    def active_conversation_id(self, tab_id: int | str | None) -> Optional[str]:
        topic_id = self.active_topic_id(tab_id)
        if not topic_id:
            return None
        return self.topic_conversation_id(tab_id, topic_id)

    def _create_topic(self, tab_key: str, seed_text: str, seed_embedding: np.ndarray) -> TopicThread:
        self._counter += 1
        topic_id = f"T{self._counter}"
        title = (seed_text or "").strip().splitlines()[0][:80] or f"Topic {self._counter}"
        embedding = seed_embedding if seed_embedding is not None and seed_embedding.any() else None
        topic = TopicThread(
            topic_id=topic_id,
            title=title,
            embedding=embedding,
            created_at=datetime.utcnow(),
        )
        self._topics.setdefault(tab_key, []).append(topic)
        return topic

    def _get_topic(self, tab_id: int | str | None, topic_id: str) -> Optional[TopicThread]:
        tab_key = self._tab_key(tab_id)
        for topic in self._topics.get(tab_key, []):
            if topic.topic_id == topic_id:
                return topic
        return None

    def _embed(self, text: str, *, command: str) -> np.ndarray:
        if not text or not text.strip():
            return self.embedder.embed_text("", command=command)
        return self.embedder.embed_text(text, command=command)

    @staticmethod
    def _tab_key(tab_id: int | str | None) -> str:
        return str(tab_id or "__default__")
