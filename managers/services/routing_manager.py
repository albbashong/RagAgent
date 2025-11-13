from __future__ import annotations

import json
from typing import Any, Dict


def _safe_json(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


class MultiSignalRouter:
    def __init__(self, run_llm_call, tool_definitions: list[Dict[str, Any]]):
        self.run_llm_call = run_llm_call
        self.tool_definitions = tool_definitions

    async def _call_router(self, payload: str, task: str) -> Dict[str, Any] | None:
        raw = await self.run_llm_call(payload, task=task, max_new_tokens=256)
        return _safe_json(raw)

    async def route(self, user_text: str, self_check: Dict[str, Any] | None) -> Dict[str, Any]:
        source = await self._call_router(user_text, "source_router") or {"source": "repo_chunks", "confidence": 0.3}
        intent = await self._call_router(user_text, "intent_router") or {"intent": "lookup", "confidence": 0.3}
        safety = await self._call_router(user_text, "safety_router") or {"override_source": "none", "override_intent": "none", "reason": ""}

        routing_context = self._aggregate(self_check, source, intent, safety)

        function_input = json.dumps(
            {
                "routing_context": routing_context,
                "functions": self.tool_definitions,
            },
            ensure_ascii=False,
        )
        function_call = await self._call_router(function_input, "function_router")
        if not function_call or not function_call.get("name") or function_call.get("name") == "none":
            function_call = self._default_function_call(routing_context, user_text)

        return {
            "source": source,
            "intent": intent,
            "safety": safety,
            "routing_context": routing_context,
            "function_call": function_call,
        }

    def _aggregate(self, self_check, source, intent, safety):
        override_source = safety.get("override_source")
        override_intent = safety.get("override_intent")
        final_source = override_source if override_source and override_source != "none" else source.get("source", "repo_chunks")
        final_intent = override_intent if override_intent and override_intent != "none" else intent.get("intent", "lookup")

        source_conf = float(source.get("confidence") or 0.0)
        intent_conf = float(intent.get("confidence") or 0.0)
        routing_confidence = round((0.6 * source_conf + 0.4 * intent_conf), 3)
        notes = []
        if override_source and override_source != "none":
            notes.append(f"source override: {safety.get('reason')}")
        if override_intent and override_intent != "none":
            notes.append(f"intent override: {safety.get('reason')}")
        if self_check and self_check.get("freshness_need") == "yes":
            notes.append("freshness required")
        if not notes:
            notes.append("auto routing")
        return {
            "final_source": final_source,
            "final_intent": final_intent,
            "routing_confidence": routing_confidence,
            "notes": "; ".join(notes),
            "freshness_needed": (self_check or {}).get("freshness_need") == "yes",
        }

    def _default_function_call(self, routing_context: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        source = routing_context.get("final_source", "repo_chunks")
        intent = routing_context.get("final_intent", "lookup")
        if source == "filesystem":
            return {"name": "search_file", "arguments": {"keyword": user_text}, "confidence": 0.25, "reason": "filesystem fallback"}
        if source == "postgres":
            if intent == "schema_view":
                return {"name": "inspect_table_columns", "arguments": {"table": "repo_meta"}, "confidence": 0.3, "reason": "schema fallback"}
            return {"name": "connect_db", "arguments": {"query": "SELECT * FROM repo_meta LIMIT 1"}, "confidence": 0.25, "reason": "postgres fallback"}
        if source == "external_web":
            return {"name": "search_web", "arguments": {"query": user_text}, "confidence": 0.3, "reason": "external fallback"}
        if source == "direct_answer":
            return {"name": "answer_direct", "arguments": {}, "confidence": 0.3, "reason": "direct fallback"}
        # repo_chunks default
        return {"name": "rag_search_chunks", "arguments": {"query": user_text}, "confidence": 0.25, "reason": "chunk fallback"}


__all__ = ["MultiSignalRouter"]
