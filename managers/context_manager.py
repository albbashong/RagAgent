class ContextManager:
    MAX_HISTORY = 10  # 최근 10개 대화만 유지
    SUMMARIZE_AFTER = 20  # 20개 이상이면 요약 실행

    def __init__(self, llm=None):
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.session_summary: dict[str, str] = {}
        self.llm = llm  # 요약용 LLMManager 인스턴스

    def add_message(self, tab_id: str | int | None, role: str, content: str):
        key = str(tab_id or "__default__")
        self.sessions.setdefault(key, []).append({"role": role, "content": content})

        # 일정 개수 초과 시 요약
        if len(self.sessions[key]) > self.SUMMARIZE_AFTER:
            self._summarize_and_trim(key)

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

    def build_prompt(self, tab_id: str | None, user_text: str) -> str:
        key = str(tab_id or "__default__")
        context = self.sessions.get(key, [])
        lines = ["### Conversation Summary ###"]
        lines.append(self.session_summary.get(key, "(No previous summary)"))
        lines.append("### Recent Messages ###")

        for i, msg in enumerate(context[-self.MAX_HISTORY:], start=1):
            lines.append(f"[{i}] {msg['role'].capitalize()}: {msg['content']}")

        lines.append("### End ###")
        lines.append(f"User: {user_text}")
        lines.append(
            "\nIf user asks about earlier messages, refer to the numbered list or the summary above."
        )
        return "\n".join(lines)
