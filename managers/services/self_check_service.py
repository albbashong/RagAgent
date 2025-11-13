from __future__ import annotations

import json
from typing import Any, Dict


class SelfCheckService:
    def __init__(self, *, llm_manager, run_llm_call):
        self.has_self_checker = "self_checker" in llm_manager.prompts
        self.run_llm_call = run_llm_call

    async def run(self, text: str) -> Dict[str, Any] | None:
        if not self.has_self_checker or not text or not text.strip():
            return None
        raw = await self.run_llm_call(text, task="self_checker", max_new_tokens=64)
        return self._parse_output(raw)

    @staticmethod
    def should_force_search(result: Dict[str, Any] | None) -> bool:
        if not result:
            return False
        return result.get("freshness_need") == "yes"

    @staticmethod
    def _parse_output(raw: str) -> Dict[str, Any] | None:
        if not raw:
            return None
        confidence = None
        freshness = None
        for line in raw.splitlines():
            clean = line.strip()
            if clean.lower().startswith("confidence:"):
                try:
                    confidence = float(clean.split(":", 1)[1].strip())
                except ValueError:
                    confidence = None
            elif clean.lower().startswith("freshness_need:"):
                value = clean.split(":", 1)[1].strip().lower()
                freshness = "yes" if value == "yes" else "no"
        if confidence is None and freshness is None:
            return None
        return {"confidence": confidence, "freshness_need": freshness}


__all__ = ["SelfCheckService"]
