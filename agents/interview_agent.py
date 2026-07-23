from __future__ import annotations

from typing import Any


class InterviewAgent:
    def __init__(self, interview_policy: list[dict[str, str]]) -> None:
        self.interview_policy = interview_policy

    def _next_question(self, state: dict[str, Any]) -> tuple[str, str]:
        followup_field = state.get("followup_field")
        if followup_field:
            for item in self.interview_policy:
                if item["field"] == followup_field:
                    return item["field"], item["question"]

        profile = state.get("applicant_profile", {})
        for item in self.interview_policy:
            if item["field"] not in profile:
                return item["field"], item["question"]

        return "general", "Please share any additional details relevant to your mortgage application."

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        field, question = self._next_question(state)
        print(f"\nInterview Agent: {question}")
        user_response = input("Applicant: ").strip()

        history = list(state.get("conversation_history", []))
        history.append({"role": "assistant", "content": question})
        history.append({"role": "user", "content": user_response})

        state["current_question"] = question
        state["current_question_field"] = field
        state["latest_user_response"] = user_response
        state["conversation_history"] = history
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        return state
