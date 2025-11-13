from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from managers.db_manager import get_connection
from managers.rag_query import RAGQueryManager


class ToolService:
    FILE_TOKEN_PATTERN = re.compile(
        r"([A-Za-z0-9_\-./]+?\.(?:ya?ml|json|py|ts|js|md|txt|c|cpp|java|rs|go|sh))"
    )

    def __init__(
        self,
        *,
        llm_manager,
        run_llm_call,
        workspace_root: Path,
        rag_manager: RAGQueryManager | None = None,
    ):
        self.llm_manager = llm_manager
        self.run_llm_call = run_llm_call
        self.workspace_root = workspace_root
        self.function_definitions: List[Dict[str, Any]] = llm_manager.prompts.get("functions", []) or []
        self.has_function_router = "function_router" in llm_manager.prompts
        self.rag_manager = rag_manager or RAGQueryManager()

    async def plan_tool(self, user_text: str) -> Dict[str, Any] | None:
        if not self.function_definitions or not self.has_function_router:
            return None
        catalog = json.dumps(self.function_definitions, ensure_ascii=False, indent=2)
        planner_prompt = (
            "다음은 사용할 수 있는 함수 목록과 스키마입니다.\n"
            f"{catalog}\n\n"
            "사용자 요청:\n"
            f"{user_text}\n\n"
            "위 요청을 해결하기 위해 호출할 최적의 함수를 JSON 한 줄로만 응답하세요. "
            '형식: {"name": "<함수명 또는 none>", "arguments": {...}, "confidence": <0.0~1.0>, "reason": "..."}\n'
            "- freshness_sensitive가 true인 함수는 신선한 정보가 필요할 때만 사용하세요.\n"
            "- freshness_sensitive가 false인 함수는 로컬 데이터나 내부 지식 기반으로 해결할 때 사용하세요."
        )
        raw = await self.run_llm_call(planner_prompt, task="function_router", max_new_tokens=256)
        parsed = self._safe_json_object(raw)
        if parsed and isinstance(parsed, dict) and parsed.get("name"):
            return parsed
        return None

    def fallback_tool(self, user_text: str) -> Dict[str, Any] | None:
        lowered = user_text.lower()
        if any(keyword in lowered for keyword in ["readme", "파일", "파일이", "파일도", "파일명", "file", "path"]):
            return self.default_rag_files_call(user_text)

        matches = self.FILE_TOKEN_PATTERN.findall(user_text)
        for token in matches:
            cleaned = token.strip().lstrip("./")
            if not cleaned:
                continue
            final_path = self._resolve_existing_path(cleaned)
            if final_path:
                return {"name": "read_file", "arguments": {"path": str(final_path)}}
        if matches:
            return {"name": "search_file", "arguments": {"keyword": matches[0]}}
        return None

    def default_rag_call(self, user_text: str) -> Dict[str, Any]:
        return {"name": "rag_search", "arguments": {"query": user_text}}

    def default_rag_files_call(self, user_text: str) -> Dict[str, Any]:
        return {"name": "rag_search_files", "arguments": {"query": user_text}}

    def is_local_context_tool(self, call: Dict[str, Any] | None) -> bool:
        if not call:
            return False
        return call.get("name") in {"read_file", "search_file"}

    def build_search_web_call(self, user_text: str) -> Dict[str, Any]:
        return {"name": "search_web", "arguments": {"query": user_text}}

    def is_answer_direct(self, call: Dict[str, Any] | None) -> bool:
        return bool(call and call.get("name") == "answer_direct")

    async def execute_tool(self, call: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any] | None]:
        name = (call.get("name") or "").strip()
        if not name or name == "none":
            raise ValueError("No executable function selected.")
        func = self._sync_functions().get(name)
        if not func:
            raise ValueError(f"Unsupported function: {name}")
        arguments = call.get("arguments") or {}
        if name == "search_file":
            arguments = await self._prepare_search_file_arguments(arguments)
        result = await asyncio.to_thread(func, arguments)
        meta = None
        text = result
        if isinstance(result, dict) and "text" in result:
            text = result["text"]
            meta = result.get("meta")
        executed = {"name": name, "arguments": arguments}
        return text, executed, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_functions(self):
        return {
            "load_files": self._execute_load_files,
            "connect_db": self._execute_connect_db,
            "inspect_table_columns": self._execute_inspect_table,
            "search_web": self._execute_search_web,
            "search_file": self._execute_search_file,
            "read_file": self._execute_read_file,
            "rag_search": self._execute_rag_search_chunks,
            "rag_search_chunks": self._execute_rag_search_chunks,
            "rag_search_files": self._execute_rag_search_files,
            "rag_search_symbols": self._execute_rag_search_symbols,
        }

    def _resolve_existing_path(self, relative: str) -> Path | None:
        candidate = (self.workspace_root / relative).resolve()
        if str(candidate).startswith(str(self.workspace_root)) and candidate.exists() and candidate.is_file():
            return candidate
        for root, _dirs, files in os.walk(self.workspace_root):
            if os.path.basename(candidate) in files:
                match = Path(root) / os.path.basename(candidate)
                if match.exists() and match.is_file():
                    return match
        return None

    def _resolve_workspace_path(self, value: str) -> Path:
        if not value:
            raise ValueError("path is required")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        if not str(candidate).startswith(str(self.workspace_root)):
            raise ValueError("Access outside workspace is not allowed")
        return candidate

    def _execute_load_files(self, arguments: Dict[str, Any]) -> str:
        target = self._resolve_workspace_path(arguments.get("path", ""))
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_dir():
            entries = sorted(p.name for p in target.iterdir())
            snippet = "\n".join(entries[:100])
            if len(entries) > 100:
                snippet += "\n... (truncated)"
            return f"[Directory Listing] {target}\n{snippet}"
        data = target.read_text(encoding="utf-8", errors="ignore")
        preview = data[:4000]
        if len(data) > 4000:
            preview += "\n... (truncated)"
        return f"[File Content] {target}\n{preview}"

    def _execute_connect_db(self, arguments: Dict[str, Any]) -> str:
        query = (arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        if not query.lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed for safety.")
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            limited = rows[:50]
            result_rows = [
                {columns[idx]: value for idx, value in enumerate(row)}
                for row in limited
            ]
            return json.dumps({"rows": result_rows, "rowCount": len(rows)}, ensure_ascii=False, indent=2)
        finally:
            conn.close()

    def _execute_inspect_table(self, arguments: Dict[str, Any]) -> str:
        table = (arguments.get("table") or "").strip()
        if not table:
            raise ValueError("table is required")
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
                (table,),
            )
            rows = cur.fetchall()
            if not rows:
                return f"[Inspect Table] No columns found for table '{table}'."
            payload = [f"{r[0]} ({r[1]})" for r in rows]
            return "[Inspect Table]\n" + "\n".join(payload)
        finally:
            conn.close()

    def _execute_search_web(self, arguments: Dict[str, Any]) -> str:
        query = arguments.get("query") or ""
        return (
            "[Search Placeholder] Web search is not available in this environment. "
            f"Intended query: {query}"
        )

    def _execute_search_file(self, arguments: Dict[str, Any]) -> str:
        keyword = (arguments.get("keyword") or "").strip()
        keyword = keyword.strip("'\"` “”‘’")
        keyword = "".join(ch for ch in keyword if ch.isalnum() or ch in {".", "_", "-", "/"})
        if not keyword:
            raise ValueError("keyword is required")
        max_results = arguments.get("max_results")
        try:
            max_results = int(max_results) if max_results is not None else 50
        except (TypeError, ValueError):
            max_results = 50
        max_results = max(1, min(max_results, 200))
        matches: List[str] = []
        for root, _dirs, files in os.walk(self.workspace_root):
            if len(matches) >= max_results:
                break
            for fname in files:
                if len(matches) >= max_results:
                    break
                if keyword.lower() in fname.lower():
                    full_path = Path(root) / fname
                    rel = full_path.relative_to(self.workspace_root)
                    matches.append(str(rel))
        if not matches:
            return f"[Search File] No files found matching '{keyword}'."
        return json.dumps({"keyword": keyword, "results": matches}, ensure_ascii=False, indent=2)

    async def _prepare_search_file_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = (arguments.get("keyword") or "").strip()
        if not raw:
            return arguments
        prompt = (
            "Extract the most relevant filename or keyword from the following text.\n"
            f"Raw text: {raw}\n"
            "Return only the keyword without punctuation."
        )
        cleaned = raw
        try:
            response = await self.run_llm_call(prompt, task="keyword_extractor", max_new_tokens=32)
            candidate = response.strip().splitlines()[0].strip().strip("'\"` “”‘’")
            candidate = "".join(ch for ch in candidate if ch.isalnum() or ch in {".", "_", "-", "/"})
            if candidate:
                cleaned = candidate
        except Exception:
            cleaned = raw
        new_args = dict(arguments)
        new_args["keyword"] = cleaned
        return new_args

    def _execute_read_file(self, arguments: Dict[str, Any]) -> str:
        target = self._resolve_workspace_path(arguments.get("path", ""))
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {target}")
        data = target.read_text(encoding="utf-8", errors="ignore")
        preview = data[:8000]
        if len(data) > 8000:
            preview += "\n... (truncated)"
        return f"[Read File] {target}\n{preview}"

    def _execute_rag_search_chunks(self, arguments: Dict[str, Any]) -> str:
        return self._format_rag_output(
            label="repo_chunks",
            query_text=arguments.get("query"),
            results=self.rag_manager.search_chunks(
                arguments.get("query"),
                top_k=arguments.get("top_k") or 5,
                repo_id=arguments.get("repo_id"),
            ),
            top_k=arguments.get("top_k") or 5,
        )

    def _execute_rag_search_files(self, arguments: Dict[str, Any]) -> str:
        return self._format_rag_output(
            label="files_meta",
            query_text=arguments.get("query"),
            results=self.rag_manager.search_files(
                arguments.get("query"),
                top_k=arguments.get("top_k") or 5,
                repo_id=arguments.get("repo_id"),
            ),
            top_k=arguments.get("top_k") or 5,
        )

    def _execute_rag_search_symbols(self, arguments: Dict[str, Any]) -> str:
        return self._format_rag_output(
            label="symbol_links",
            query_text=arguments.get("query"),
            results=self.rag_manager.search_symbols(
                arguments.get("query"),
                top_k=arguments.get("top_k") or 5,
                repo_id=arguments.get("repo_id"),
            ),
            top_k=arguments.get("top_k") or 5,
        )

    def _format_rag_output(self, *, label: str, query_text: str, results, top_k: int) -> str:
        query_text = (query_text or "").strip()
        if not query_text:
            raise ValueError("query is required")
        payload = [item.to_dict() for item in results]
        lines = [
            f"[RAG Search Results - {label}]",
            f"Query: {query_text}",
            f"Matches: {len(payload)} (top_k={top_k})",
        ]
        if results:
            top_result = results[0]
            snippet = top_result.content.strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."
            lines.append("\n[Top Match]")
            lines.append(
                f"file_path={top_result.file_path} repo_id={top_result.repo_id} score={top_result.score:.4f}"
            )
            if top_result.semantic_scope:
                lines.append(f"scope: {top_result.semantic_scope}")
            lines.append(f"content: {snippet}")
            suggestions = self._build_suggestions(query_text, top_result)
            if suggestions:
                lines.append("\n[Suggested Actions]")
                for idx, suggestion in enumerate(suggestions, start=1):
                    lines.append(f"{idx}. {suggestion}")
        else:
            lines.append("\n[Top Match]\n관련 결과를 찾지 못했습니다.")
        return {"text": "\n".join(lines), "meta": {"match_count": len(payload)}}

    def _build_suggestions(self, query_text: str, top_result) -> List[str]:
        file_hint = top_result.file_path if top_result else "해당 위치"
        return [
            f"{file_hint} 관련 코드를 최신 PyTorch 스타일로 리팩터링해 드릴까요?",
            "이 코드/컨텐츠가 수행하는 동작을 단계별로 설명해 드릴까요?",
            "연관된 파일이나 추가 예제를 찾아 간단히 요약해 드릴까요?",
        ]

    @staticmethod
    def _safe_json_object(text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None


__all__ = ["ToolService"]
