from __future__ import annotations

from typing import Any

from core import terminal_ui
from core.schemas import FIELD_LABELS


class InterviewAgent:
    """
    Conducts the intake as a natural back-and-forth conversation rather than a
    rigid form. The LLM decides how to phrase each turn (greeting, acknowledging
    the applicant's last answer, asking about what's still missing). If the LLM
    is unavailable, it falls back to the plain scripted question for that field.
    """

    def __init__(
        self,
        interview_policy: list[dict[str, str]],
        system_prompt: str,
        llm_client: Any = None,
    ) -> None:
        self.interview_policy = interview_policy
        self.system_prompt = system_prompt.strip()
        self.llm_client = llm_client

    def _missing_fields(self, state: dict[str, Any]) -> tuple[list[str], str | None]:
        profile = state.get("applicant_profile", {})
        followup_field = state.get("followup_field")
        order = [item["field"] for item in self.interview_policy]
        missing = [f for f in order if f not in profile]
        if followup_field and followup_field not in missing:
            missing = [followup_field] + missing
        return missing, followup_field

    def _static_question(self, field: str | None) -> str:
        for item in self.interview_policy:
            if item["field"] == field:
                return item["question"]
        return "Is there anything else you'd like to add about your situation?"

    def _generate_conversational_question(
        self,
        state: dict[str, Any],
        missing: list[str],
        followup_field: str | None,
        is_first_turn: bool,
    ) -> str:
        if not self.llm_client:
            return ""

        history = [m for m in state.get("conversation_history", []) if m.get("role") != "system"]
        field_context = ", ".join(FIELD_LABELS.get(f, f) for f in missing[:3])

        if followup_field:
            nudge = (
                f"The applicant's last answer about {FIELD_LABELS.get(followup_field, followup_field)} "
                "wasn't clear or usable. Warmly ask them to clarify just that one thing, in plain "
                "everyday language — don't mention 'validation' or technical field names."
            )
        elif is_first_turn:
            nudge = (
                "This is the very start of the conversation. Greet the applicant warmly in one short "
                "sentence, briefly explain you'll chat with them to understand their situation for a "
                f"mortgage pre-check, then naturally ask about: {field_context}."
            )
        else:
            nudge = (
                f"Briefly and naturally acknowledge what they just told you (one short phrase), then ask "
                f"about: {field_context}. Ask about only one topic at a time, unless two are closely "
                "related (e.g. property price and loan amount)."
            )

        try:
            return self.llm_client.generate_reply(
                self.system_prompt,
                history + [{"role": "system", "content": nudge}],
            )
        except Exception:
            return ""

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        missing, followup_field = self._missing_fields(state)
        target_field = (followup_field if followup_field in missing else missing[0]) if missing else None

        history = list(state.get("conversation_history", []))
        is_first_turn = not any(m.get("role") == "assistant" for m in history)

        question = self._generate_conversational_question(state, missing, followup_field, is_first_turn)
        if not question:
            base = self._static_question(target_field)
            question = (
                f"Hi! I'm here to help figure out your mortgage eligibility. {base}"
                if is_first_turn else base
            )

        terminal_ui.print_agent_message(question, is_first_turn=is_first_turn)
        user_response = terminal_ui.get_answer_prompt()
        terminal_ui.print_thinking()

        if not history and self.system_prompt:
            history.append({"role": "system", "content": self.system_prompt})
        history.append({"role": "assistant", "content": question})
        history.append({"role": "user", "content": user_response})

        state["current_question"] = question
        state["current_question_field"] = target_field or "general"
        state["latest_user_response"] = user_response
        state["conversation_history"] = history
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        return state