from __future__ import annotations

import logging
import re
from typing import Any

from core import terminal_ui
from core.schemas import FIELD_LABELS

logger = logging.getLogger(__name__)


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
        greeting_text: str | None = None,
    ) -> None:
        self.interview_policy = interview_policy
        self.system_prompt = system_prompt.strip()
        self.llm_client = llm_client
        self.greeting_text = (greeting_text or "").strip()
        self.STOP_COMMANDS = {"stop", "end", "/stop", "/end"}
        self.AFFIRMATIVE = {"yes", "y", "sure", "ok", "okay", "please"}
        self.NEGATIVE = {"no", "n", "not now", "nope"}
        self.FINALIZE_PATTERNS = [
            re.compile(
                r"^no(?:\s*(?:,?\s*(?:that'?s\s+all(?:\s+the\s+information\s+i\s+have)?|that\s+is\s+all(?:\s+the\s+information\s+i\s+have)?|all\s+i\s+have|that's\s+all|that\s+is\s+all|thanks?|thank\s+you|not\s+right\s+now|that's\s+it|that\s+is\s+it))?)?\s*[.!?]*$",
                re.IGNORECASE,
            ),
        ]

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

    def _detect_greeting(self, text: str | None) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if any(token in lowered for token in ["income", "credit", "debt", "loan", "property", "job", "employment", "score"]):
            return False
        return any(
            lowered in {"hi", "hello", "hey", "hi there", "hey there", "hello there"}
            or lowered.startswith(token)
            for token in ["hi ", "hello ", "hey ", "good morning", "good afternoon", "good evening"]
        )

    def _detect_stop_intent(self, text: str | None) -> bool:
        cleaned = (text or "").strip().lower()
        if not cleaned:
            return False
        if cleaned in self.STOP_COMMANDS:
            return True
        stop_phrases = [
            "i don't want to continue",
            "i do not want to continue",
            "i don't want to answer anymore",
            "i do not want to answer anymore",
            "please stop",
            "don't contact me anymore",
            "do not contact me anymore",
            "leave me alone",
            "i'm not interested anymore",
            "i am not interested anymore",
            "i don't want to continue this conversation",
            "i do not want to continue this conversation",
            "no longer interested",
            "stop contacting me",
            "end this conversation",
        ]
        return any(phrase in cleaned for phrase in stop_phrases)

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

        # If the decision agent has already offered early termination, confirm it before asking
        # another generic followup question. This keeps the UX stable when multiple fields were
        # provided in one turn and a likely conclusion is already available.
        if state.get("offer_early_termination") and not state.get("early_offered_already"):
            confirm_q = (
                "I can already reach a likely conclusion based on what you've provided. "
                "Would you like me to finalize now? (yes/no)"
            )
            terminal_ui.print_agent_message(confirm_q, is_first_turn=is_first_turn)
            confirm_resp = terminal_ui.get_answer_prompt().strip()
            terminal_ui.print_thinking()

            if not history and self.system_prompt:
                history.append({"role": "system", "content": self.system_prompt})
            history.append({"role": "assistant", "content": confirm_q})
            history.append({"role": "user", "content": confirm_resp})

            state["conversation_history"] = history
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["early_offered_already"] = True
            state["current_question"] = confirm_q
            state["latest_user_response"] = confirm_resp

            is_confirm_finalize = any(pattern.match(confirm_resp.strip()) for pattern in self.FINALIZE_PATTERNS)
            resp_clean = (confirm_resp or "").strip().lower()
            if is_confirm_finalize:
                state["user_requested_finalize"] = True
                state["needs_followup"] = False
                state["followup_field"] = None
                state["offer_early_termination"] = False
                return state
            if resp_clean in self.AFFIRMATIVE:
                state["user_confirmed_early_end"] = True
                state["needs_followup"] = False
            else:
                state["offer_early_termination"] = False
                state["user_confirmed_early_end"] = False

            return state

        question = self._generate_conversational_question(state, missing, followup_field, is_first_turn)
        if not question:
            base = self._static_question(target_field)
            question = (
                f"Hi! I'm here to help figure out your mortgage eligibility. {base}"
                if is_first_turn else base
            )
        # Prepend configured greeting on the very first assistant turn if provided
        if is_first_turn and self.greeting_text:
            # avoid duplicating if the LLM already included a greeting
            if not question.lower().startswith(self.greeting_text.lower()):
                question = f"{self.greeting_text} {question}"

        terminal_ui.print_agent_message(question, is_first_turn=is_first_turn)
        user_response = terminal_ui.get_answer_prompt()
        terminal_ui.print_thinking()

        cleaned = (user_response or "").strip().lower()
        is_stop_command = self._detect_stop_intent(user_response)
        is_finalize_phrase = any(pattern.match(user_response.strip()) for pattern in self.FINALIZE_PATTERNS)
        is_greeting = self._detect_greeting(user_response)
        logger.debug(
            "User response=%r cleaned=%r is_stop=%s is_finalize=%s is_greeting=%s",
            user_response,
            cleaned,
            is_stop_command,
            is_finalize_phrase,
            is_greeting,
        )

        # ----- Stop command detection (user-initiated end) -----
        if is_stop_command:
            # record final user message and mark session stopped
            if not history and self.system_prompt:
                history.append({"role": "system", "content": self.system_prompt})
            history.append({"role": "assistant", "content": question})
            history.append({"role": "user", "content": user_response})

            state["conversation_history"] = history
            state["current_question"] = question
            state["latest_user_response"] = user_response
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["user_requested_stop"] = True
            state["skip_extraction"] = True
            state["needs_followup"] = False
            state["followup_field"] = None
            state["decision_status"] = "Stopped by User"
            state["decision_summary"] = "User ended the conversation."
            state["lead_step"] = "stop"
            state["summary"] = "User ended the conversation."
            state["qualification_category"] = "stopped"
            state["conversation_status"] = "stopped"
            state["session_tags"] = ["ciap-stopped"]
            state["final_report"] = {
                "status": "Stopped by User",
                "summary": "User ended the conversation.",
                "stopped_by_user": True,
            }
            return state

        if is_greeting:
            if not history and self.system_prompt:
                history.append({"role": "system", "content": self.system_prompt})
            history.append({"role": "assistant", "content": question})
            history.append({"role": "user", "content": user_response})

            state["conversation_history"] = history
            state["current_question"] = question
            state["current_question_field"] = target_field or "general"
            state["latest_user_response"] = user_response
            state["conversation_history"] = history
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["greeting_detected"] = True
            state["skip_extraction"] = True
            state["needs_followup"] = True
            state["followup_field"] = target_field
            state["lead_step"] = "greeting"
            state["summary"] = "User greeted the assistant."
            state["qualification_category"] = state.get("qualification_category") or "in_progress"
            state["conversation_status"] = "in_progress"
            return state

        # ----- User finalize-at-current-state handling -----
        if is_finalize_phrase:
            if not history and self.system_prompt:
                history.append({"role": "system", "content": self.system_prompt})
            history.append({"role": "assistant", "content": question})
            history.append({"role": "user", "content": user_response})

            state["conversation_history"] = history
            state["current_question"] = question
            state["current_question_field"] = target_field or "general"
            state["latest_user_response"] = user_response
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["user_requested_finalize"] = True
            state["needs_followup"] = False
            state["followup_field"] = None
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = "User indicated no more information is available."
            return state

        # ----- Early-termination confirmation handling -----
        # If the decision agent offered early termination, ask for confirmation.
        if state.get("offer_early_termination") and not state.get("early_offered_already"):
            # ask a short confirmation instead of the usual followup
            confirm_q = (
                "I can already reach a likely conclusion based on what you've provided. "
                "Would you like me to finalize now? (yes/no)"
            )
            terminal_ui.print_agent_message(confirm_q)
            confirm_resp = terminal_ui.get_answer_prompt().strip()
            terminal_ui.print_thinking()
            # append assistant prompt and user response
            if not history and self.system_prompt:
                history.append({"role": "system", "content": self.system_prompt})
            history.append({"role": "assistant", "content": confirm_q})
            history.append({"role": "user", "content": confirm_resp})

            state["conversation_history"] = history
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["early_offered_already"] = True

            resp_clean = (confirm_resp or "").strip().lower()
            if resp_clean in self.AFFIRMATIVE:
                state["user_confirmed_early_end"] = True
                state["needs_followup"] = False
            else:
                # user declined early termination; continue normal flow
                state["offer_early_termination"] = False
                state["user_confirmed_early_end"] = False

            return state

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