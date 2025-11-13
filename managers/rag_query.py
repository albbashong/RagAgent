from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from managers.embedding import EmbeddingManager
from managers.db_manager import get_connection


def _vector_to_pg_string(vector) -> str:
    values = ",".join(f"{float(v):.8f}" for v in vector)
    return f"[{values}]"


@dataclass
class RAGQueryResult:
    table: str
    id: int
    repo_id: Optional[int]
    file_id: Optional[int]
    file_path: Optional[str]
    semantic_scope: Optional[str]
    hierarchical_context: Optional[str]
    content: str
    extras: Dict[str, Any]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "id": self.id,
            "repo_id": self.repo_id,
            "file_id": self.file_id,
            "file_path": self.file_path,
            "semantic_scope": self.semantic_scope,
            "hierarchical_context": self.hierarchical_context,
            "content": self.content,
            "extras": self.extras,
            "score": self.score,
        }


class RAGQueryManager:
    def __init__(self, embedder: EmbeddingManager | None = None):
        self.embedder = embedder or EmbeddingManager()

    def search_chunks(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        repo_id: Optional[int] = None,
    ) -> List[RAGQueryResult]:
        vector_literal, top_k = self._prepare_query(query_text, top_k)
        conn = get_connection()
        try:
            cur = conn.cursor()
            sql = (
                "SELECT id, repo_id, file_id, file_path, semantic_scope,"
                " hierarchical_context, content, embedding <=> %s AS score "
                "FROM repo_chunks "
                "WHERE embedding IS NOT NULL"
            )
            params: List[Any] = [vector_literal]
            if repo_id is not None:
                sql += " AND repo_id = %s"
                params.append(int(repo_id))
            sql += " ORDER BY embedding <=> %s LIMIT %s"
            params.extend([vector_literal, top_k])
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [
                RAGQueryResult(
                    table="repo_chunks",
                    id=row[0],
                    repo_id=row[1],
                    file_id=row[2],
                    file_path=row[3],
                    semantic_scope=row[4],
                    hierarchical_context=row[5],
                    content=row[6],
                    extras={},
                    score=float(row[7]),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def search_files(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        repo_id: Optional[int] = None,
    ) -> List[RAGQueryResult]:
        vector_literal, top_k = self._prepare_query(query_text, top_k)
        conn = get_connection()
        try:
            cur = conn.cursor()
            sql = (
                "SELECT id, repo_id, file_path, file_type, summary, embedding <=> %s AS score "
                "FROM files_meta "
                "WHERE embedding IS NOT NULL"
            )
            params: List[Any] = [vector_literal]
            if repo_id is not None:
                sql += " AND repo_id = %s"
                params.append(int(repo_id))
            sql += " ORDER BY embedding <=> %s LIMIT %s"
            params.extend([vector_literal, top_k])
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [
                RAGQueryResult(
                    table="files_meta",
                    id=row[0],
                    repo_id=row[1],
                    file_id=None,
                    file_path=row[2],
                    semantic_scope=row[3],
                    hierarchical_context=None,
                    content=row[4] or "",
                    extras={"file_type": row[3]},
                    score=float(row[5]),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def search_symbols(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        repo_id: Optional[int] = None,
    ) -> List[RAGQueryResult]:
        term = (query_text or "").strip()
        top_k = max(1, min(int(top_k or 5), 50))
        conn = get_connection()
        try:
            cur = conn.cursor()
            sql = (
                "SELECT id, repo_id, source_symbol, target_symbol, relation_type, file_path, created_at "
                "FROM symbol_links "
                "WHERE source_symbol ILIKE %s OR target_symbol ILIKE %s"
            )
            params: List[Any] = [f"%{term}%", f"%{term}%"]
            if repo_id is not None:
                sql += " AND repo_id = %s"
                params.append(int(repo_id))
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(top_k)
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [
                RAGQueryResult(
                    table="symbol_links",
                    id=row[0],
                    repo_id=row[1],
                    file_id=None,
                    file_path=row[5],
                    semantic_scope=row[2],
                    hierarchical_context=row[3],
                    content=f"{row[2]} -> {row[3]} ({row[4]})",
                    extras={"relation_type": row[4]},
                    score=0.0,
                )
                for row in rows
            ]
        finally:
            conn.close()

    def _prepare_query(self, query_text: str, top_k: int) -> tuple[str, int]:
        vector = self.embedder.embed_text(query_text, command="query")
        vector_literal = _vector_to_pg_string(vector)
        top_k = max(1, min(int(top_k or 5), 50))
        return vector_literal, top_k


__all__ = ["RAGQueryManager", "RAGQueryResult"]
