from __future__ import annotations

import logging
import re
from typing import Any

from core import terminal_ui
from core.schemas import FIELD_LABELS

logger = logging.getLogger(__name__)


class InterviewAgent:
    """
    Conversational interview agent.

    IMPORTANT DESIGN RULE:
        Python/state decides WHICH field should be asked next.
        The LLM decides HOW that field should be asked.

    This prevents the LLM from changing the interview order.
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

        self.STOP_COMMANDS = {
            "stop",
            "end",
            "/stop",
            "/end",
        }

        self.AFFIRMATIVE = {
            "yes",
            "y",
            "ye",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "please",
            "go ahead",
        }

        self.NEGATIVE = {
            "no",
            "n",
            "not now",
            "nope",
        }

        self.FINALIZE_PATTERNS = [
            re.compile(
                r"^no(?:\s*(?:,?\s*(?:"
                r"that'?s\s+all(?:\s+the\s+information\s+i\s+have)?|"
                r"that\s+is\s+all(?:\s+the\s+information\s+i\s+have)?|"
                r"all\s+i\s+have|"
                r"that's\s+all|"
                r"that\s+is\s+all|"
                r"thanks?|"
                r"thank\s+you|"
                r"not\s+right\s+now|"
                r"that's\s+it|"
                r"that\s+is\s+it"
                r"))?)?\s*[.!?]*$",
                re.IGNORECASE,
            ),
        ]

    # ================================================================
    # FIELD MANAGEMENT
    # ================================================================

    def _missing_fields(
        self,
        state: dict[str, Any],
    ) -> tuple[list[str], str | None]:

        profile = state.get(
            "applicant_profile",
            {},
        ) or {}

        followup_field = state.get(
            "followup_field"
        )

        # The interview policy is the ONLY source for the interview order.
        policy_order = [
            item["field"]
            for item in self.interview_policy
        ]

        # A field is missing when it has not yet been successfully
        # stored in the applicant profile.
        skipped_fields = set(state.get("skipped_fields", []))
        missing = [
            field
            for field in policy_order
            if field not in profile and field not in skipped_fields
        ]

        # If a previous answer requires clarification,
        # that field gets priority.
        if followup_field:
            missing = [
                followup_field,
                *[
                    field
                    for field in missing
                    if field != followup_field
                ],
            ]

        return missing, followup_field

    def _detect_skip_intent(self, text: str | None) -> bool:
        """Recognize a request to defer the current question."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
        refusal_patterns = (
            r"i (?:don'?t|do not) want to (?:tell|say|share)(?: (?:it|you|u))?",
            r"(?:i )?(?:don'?t|do not|can'?t|cannot) remember",
            r"(?:i have )?no idea",
            r"not (?:sure|remember)",
            r"no[ ,_-]*pass",
        )
        if any(re.fullmatch(pattern + r"[.!?]*", cleaned) for pattern in refusal_patterns):
            return True
        return bool(
            re.fullmatch(
                r"(?:skip(?: this)?(?: question)?|pass|next(?: question| ques)?|"
                r"move (?:to )?(?:the )?next(?: question| ques| quest)?|"
                r"ask (?:the )?next(?: question| ques)?|i (?:don'?t|do not) (?:know|want to answer))"
                r"[.!?]*",
                cleaned,
            )
        )

    def _detect_finalize_intent(
        self,
        text: str | None,
        question: str,
        target_field: str | None,
    ) -> bool:
        """Treat 'no' as an answer unless the conversation is actually closing."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not cleaned:
            return False

        explicit_closing = (
            "that's all",
            "that is all",
            "all i have",
            "that's it",
            "that is it",
            "no more information",
            "nothing else",
        )
        if any(phrase in cleaned for phrase in explicit_closing):
            return True

        closing_question = target_field is None and bool(
            re.search(r"anything else|like to add|more information", question, re.IGNORECASE)
        )
        return cleaned in self.NEGATIVE and closing_question

    def _static_question(
        self,
        field: str | None,
    ) -> str:

        for item in self.interview_policy:
            if item["field"] == field:
                return item["question"]

        return (
            "Is there anything else you'd like to add "
            "about your situation?"
        )

    def _fallback_question(
        self,
        field: str | None,
        state: dict[str, Any],
        is_first_turn: bool,
    ) -> str:
        """Produce a natural question when conversational generation is unavailable."""
        latest = str(state.get("latest_user_response", "")).strip().lower()

        if field == "down_payment" and not re.search(r"\d", latest) and re.search(
            r"\b(?:yes|yeah|yep|it is|is a)\b.*\b(?:hurdle|problem|issue|obstacle)\b",
            latest,
        ):
            return (
                "I understand — coming up with the down payment can be difficult. "
                "About how much could you put down right now?"
            )

        if field == "down_payment" and re.search(
            r"\b(?:no|not really|nope)\b.*\b(?:hurdle|problem|issue|obstacle)\b",
            latest,
        ):
            return (
                "Got it — so the down payment itself isn't your main obstacle. "
                "Roughly how much would you be able to put down?"
            )

        question = self._static_question(field)
        if is_first_turn:
            return f"Hi! I'm here to help with your mortgage pre-check. {question}"

        profile = state.get("applicant_profile", {}) or {}
        if field == "credit_score" and "down_payment" in profile:
            amount = float(profile["down_payment"])
            shown = f"{amount:,.0f}"
            return f"I've noted {shown} for the down payment. Do you know your approximate credit score?"
        if field == "employment_status" and "credit_score" in profile:
            qualifier = "rough estimate" if re.search(r"\b(?:maybe|about|around|roughly|not sure|no sure)\b", latest) else "score"
            return f"That's helpful — I've noted {int(profile['credit_score'])} as a {qualifier}. Are you currently employed, self-employed, or between jobs?"
        if field == "employment_years" and profile.get("employment_status"):
            return f"Got it, you're {profile['employment_status']}. How long have you been in your current job or business?"
        if field == "annual_income" and "employment_years" in profile:
            years = float(profile["employment_years"])
            shown = f"{years:g} year" + ("" if years == 1 else "s")
            return f"Thanks — I've noted {shown} in your current work. Roughly what do you earn in a year before tax?"
        if field == "total_savings" and "annual_income" in profile:
            return "Thanks, that gives me a clearer picture. Separate from the down payment, roughly how much do you have in savings?"

        return "Thanks, that helps. " + question

    def _validation_followup_question(
        self,
        field: str | None,
        state: dict[str, Any],
    ) -> str:
        """Explain a rejected value before asking for a corrected answer."""
        issues = " ".join(
            str(issue).lower()
            for issue in state.get("last_extraction", {}).get("issues", [])
        )
        prompts = {
            "credit_score": (
                "A credit score should be a whole number between 300 and 850. "
                "What is your score, or your best estimate?"
            ),
            "down_payment": (
                "I need a non-negative amount for the down payment. "
                "Roughly how much could you put down?"
            ),
            "employment_years": (
                "The number of years can't be negative. "
                "About how long have you been in your current job or business?"
            ),
            "annual_income": (
                "I need an annual income greater than zero. "
                "What is your approximate yearly income before tax?"
            ),
            "employment_status": (
                "I didn't quite catch your work status. "
                "Are you employed, self-employed, or currently between jobs?"
            ),
            "monthly_debt": (
                "That's okay if you don't remember the exact amount. "
                "What is your best estimate of the total you pay toward debts each month?"
            ),
            "total_savings": (
                "Savings can't be a negative amount. "
                "Roughly how much do you currently have saved?"
            ),
        }
        if field == "employment_years" and "ambiguous" in issues:
            answer = str(state.get("latest_user_response", "")).strip()
            return (
                "I want to make sure I record that correctly. "
                f"When you say '{answer}', do you mean a fraction of a year or a range of years?"
            )
        if field and field in issues:
            return prompts.get(field, "That value doesn't look valid. Could you try again?")
        return ""

    def _clarifying_question(self, field: str | None) -> str:
        """Give a concrete answer to a field question without repeating it verbatim."""
        examples = {
            "down_payment": "I mean the amount you could pay upfront toward the home price — even a rough amount is helpful.",
            "credit_score": "I mean the three-digit score from your credit report; an approximate range is fine if you do not know the exact number.",
            "employment_status": "I just need to know whether you are currently employed, self-employed, or not working — retirement or pension income counts as not working for this check.",
            "employment_years": "I mean how long you have been in your current job or business; a rough number of years is fine.",
            "annual_income": "I mean your total yearly income before tax; a rough annual amount is fine.",
            "total_savings": "I mean all savings you have set aside, separate from the amount you plan to use as a down payment.",
        }
        return examples.get(field, "Could you tell me a little more about that?")

    def _detect_clarifying_question(self, text: str | None) -> bool:
        """Recognize a request to explain the current question, not an answer."""
        cleaned = (text or "").strip().lower()
        if not cleaned or len(cleaned) > 120:
            return False
        patterns = (
            r"^(what|which) (?:do you mean|range|amount|one)\b",
            r"^(?:like )?what\??$",
            r"^how (?:much|many|do you mean)\b",
            r"^(?:can|could) you (?:explain|clarify)\b",
            r"^in which range\b",
            r"^what (?:kind|sort)\b",
            r"^i(?:'m| am) not sure what you mean\b",
        )
        return any(re.search(pattern, cleaned) for pattern in patterns)

    # ================================================================
    # GREETING / STOP DETECTION
    # ================================================================

    def _detect_greeting(
        self,
        text: str | None,
    ) -> bool:

        cleaned = (text or "").strip()

        if not cleaned:
            return False

        lowered = cleaned.lower()

        # Do not classify mortgage answers as greetings.
        mortgage_terms = [
            "income",
            "credit",
            "debt",
            "loan",
            "property",
            "job",
            "employment",
            "score",
            "down payment",
        ]

        if any(
            token in lowered
            for token in mortgage_terms
        ):
            return False

        greetings = {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hey there",
            "hello there",
        }

        if lowered in greetings:
            return True

        greeting_prefixes = [
            "hi ",
            "hello ",
            "hey ",
            "good morning",
            "good afternoon",
            "good evening",
        ]

        return any(
            lowered.startswith(prefix)
            for prefix in greeting_prefixes
        )

    def _detect_stop_intent(
        self,
        text: str | None,
    ) -> bool:

        cleaned = (
            text or ""
        ).strip().lower()

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

        return any(
            phrase in cleaned
            for phrase in stop_phrases
        )

    # ================================================================
    # CONVERSATIONAL QUESTION GENERATION
    # ================================================================

    def _sanitize_generated_response(
        self,
        response: str,
        state: dict[str, Any],
        is_first_turn: bool,
    ) -> str:
        """Enforce factual and one-question boundaries on model wording."""
        latest = str(state.get("latest_user_response", ""))
        if not re.search(r"[$₹£€]", latest):
            response = re.sub(r"[$₹£€]", "", response)

        if is_first_turn and re.search(
            r"\b(?:buying|purchasing)\b.*\b(?:alone|own|someone|partner|jointly)\b",
            response,
            re.IGNORECASE,
        ):
            return (
                "Hi there! I'm Alex, and I'm here to help with your mortgage pre-check. "
                "Will this be your first home?"
            )

        # Keep only the final question when the model produces multiple ones.
        # Earlier question sentences are conversational extras; the last one
        # is the application-selected mortgage topic.
        if response.count("?") > 1:
            parts = re.split(r"(?<=\?)\s+", response)
            question_indexes = [index for index, part in enumerate(parts) if "?" in part]
            keep_question = question_indexes[-1]
            parts = [
                part
                for index, part in enumerate(parts)
                if "?" not in part or index == keep_question
            ]
            response = " ".join(parts)

        return response.strip()

    def _response_matches_target(
        self,
        response: str,
        target_field: str | None,
        is_first_turn: bool,
    ) -> bool:
        """Reject model questions that drift away from the selected topic."""
        text = response.lower()
        if "?" not in response:
            return False
        if is_first_turn:
            return "first home" in text
        patterns = {
            "down_payment": r"\b(?:down payment|put down|upfront)\b",
            "credit_score": r"\b(?:credit|score)\b",
            "employment_status": r"\b(?:employ|working|work status|job status|between jobs)\b",
            "employment_years": r"\b(?:how long|years?|months?|job|business)\b",
            "annual_income": r"\b(?:annual income|income|earn|make each year|yearly)\b",
            "total_savings": r"\b(?:savings?|saved)\b",
            "monthly_debt": r"\b(?:debt|monthly payments?|repayments?)\b",
            "property_value": r"\b(?:property|home|house|purchase price)\b",
            "requested_loan_amount": r"\b(?:loan|borrow)\b",
        }
        pattern = patterns.get(target_field or "")
        return bool(pattern and re.search(pattern, text))

    def _generate_conversational_question(
        self,
        state: dict[str, Any],
        missing: list[str],
        followup_field: str | None,
        is_first_turn: bool,
    ) -> str:

        if not self.llm_client:
            return ""

        history = [
            message
            for message in state.get(
                "conversation_history",
                [],
            )
            if message.get("role") != "system"
        ]

        # ============================================================
        # CRITICAL:
        # Python selects the target field.
        # The LLM is NOT allowed to select another field.
        # ============================================================

        target_field = (
            followup_field
            if followup_field in missing
            else (
                missing[0]
                if missing
                else None
            )
        )

        if not target_field:
            return ""

        field_label = FIELD_LABELS.get(
            target_field,
            target_field,
        )

        asked_fields = list(
            state.get(
                "asked_fields",
                [],
            )
        )

        answered_fields = list(
            state.get(
                "answered_fields",
                [],
            )
        )

        question_history = list(
            state.get(
                "question_history",
                [],
            )
        )

        # ============================================================
        # BUILD STRICT INSTRUCTION
        # ============================================================

        if followup_field:
            clarification_context = state.get("clarification_context", "")
            instruction = f"""
The applicant needs clarification about exactly this topic:

TARGET FIELD:
{target_field}

TARGET TOPIC:
{field_label}

Ask ONLY about this topic.

Do not move to another topic.
Do not introduce another field.
Do not ask a second unrelated question.

Acknowledge the previous answer briefly and naturally,
then ask for the missing clarification.

If the applicant asked what the question means, answer that question directly
before asking for the value. Their clarification request was: {clarification_context}
"""

        elif is_first_turn:
            instruction = """
This is the opening of the mortgage conversation.

Introduce yourself naturally as Alex.

Then ask ONE short warm opening question about whether
this is the applicant's first home.

Do not ask about income, credit score, debt, employment,
loan amount, property value, or down payment in the opening.

The opening question is conversational context and is
separate from the required eligibility fields.
"""

        else:
            instruction = f"""
The application has already selected the next required topic.

TARGET FIELD:
{target_field}

TARGET TOPIC:
{field_label}

YOU MUST ASK ABOUT THIS TARGET FIELD.

The target field has been selected by the application.
You are NOT allowed to select a different field.

Your job is ONLY to make the question sound natural,
warm, concise, and conversational.

Do NOT:
- skip this field
- jump to a later field
- invent a new field
- ask about annual income if the target is down payment
- ask about debt if the target is credit score
- ask multiple unrelated questions
- repeat an already answered field

You may briefly acknowledge the applicant's latest answer,
then ask ONE question about the TARGET FIELD.
"""

        memory = f"""
CONVERSATION MEMORY

Already answered fields:
{answered_fields}

Previously asked fields:
{asked_fields}

Recent question history:
{question_history[-5:]}

CURRENT TARGET FIELD:
{target_field}

CURRENT TARGET TOPIC:
{field_label}
"""

        strict_rules = """
STRICT RULES:

1. Ask exactly ONE primary question.
2. The CURRENT TARGET FIELD is authoritative.
3. Never change the target field.
4. Never skip the target field.
5. Never invent a required field.
6. Never repeat an already answered topic.
7. Keep the response short.
8. Use natural conversational language.
9. Use a brief acknowledgement when appropriate.
10. Do not mention fields, schemas, agents, state, policies,
    validation, or technical implementation.
11. Do not calculate eligibility.
12. Do not promise unsupported actions.
"""

        prompt = (
            instruction
            + "\n"
            + memory
            + "\n"
            + strict_rules
        )

        try:
            response = self.llm_client.generate_reply(
                self.system_prompt,
                history
                + [
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ],
            )

            response = (
                response or ""
            ).strip()

            # --------------------------------------------------------
            # Safety check:
            # Never accept an empty LLM response.
            # --------------------------------------------------------

            if not response:
                return ""

            response = self._sanitize_generated_response(response, state, is_first_turn)
            if not self._response_matches_target(response, target_field, is_first_turn):
                logger.debug(
                    "Discarding generated question that drifted from target field %r",
                    target_field,
                )
                return ""
            return response

        except Exception:
            logger.debug(
                "Conversational question generation failed.",
                exc_info=True,
            )
            return ""

    # ================================================================
    # QUESTION TRACKING
    # ================================================================

    def _track_question(
        self,
        state: dict[str, Any],
        field: str | None,
        question: str,
    ) -> None:

        if not field:
            return

        asked_fields = list(
            state.get(
                "asked_fields",
                [],
            )
        )

        if field not in asked_fields:
            asked_fields.append(field)

        state["asked_fields"] = asked_fields

        question_history = list(
            state.get(
                "question_history",
                [],
            )
        )

        question_history.append(
            {
                "field": field,
                "question": question,
            }
        )

        # Keep only recent questions.
        state["question_history"] = (
            question_history[-20:]
        )

        state["current_question_field"] = field

    # ================================================================
    # MAIN AGENT
    # ================================================================

    def __call__(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:

        missing, followup_field = (
            self._missing_fields(state)
        )

        # ============================================================
        # TARGET FIELD IS SELECTED HERE — NOT BY THE LLM
        # ============================================================

        target_field = (
            followup_field
            if followup_field in missing
            else (
                missing[0]
                if missing
                else None
            )
        )

        history = list(
            state.get(
                "conversation_history",
                [],
            )
        )

        is_first_turn = (
            not state.get("applicant_profile")
            and not any(
                message.get("role") == "assistant"
                for message in history
            )
        )

        # ============================================================
        # EARLY TERMINATION
        # ============================================================

        if (
            state.get(
                "offer_early_termination"
            )
            and not state.get(
                "early_offered_already"
            )
        ):

            confirm_q = (
                "I have enough information to complete your pre-check. "
                "Would you like me to show you the result now?"
            )

            terminal_ui.print_agent_message(
                confirm_q,
                is_first_turn=is_first_turn,
            )

            confirm_resp = (
                terminal_ui
                .get_answer_prompt()
                .strip()
            )

            terminal_ui.print_thinking()

            if not history and self.system_prompt:
                history.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )

            history.append(
                {
                    "role": "assistant",
                    "content": confirm_q,
                }
            )

            history.append(
                {
                    "role": "user",
                    "content": confirm_resp,
                }
            )

            state["conversation_history"] = history
            state["turn_count"] = (
                int(
                    state.get(
                        "turn_count",
                        0,
                    )
                )
                + 1
            )

            state["early_offered_already"] = True
            state["current_question"] = confirm_q
            state["latest_user_response"] = confirm_resp

            if (
                confirm_resp.lower()
                in self.AFFIRMATIVE
            ):
                state[
                    "user_confirmed_early_end"
                ] = True
                state["needs_followup"] = False

            elif (
                confirm_resp.lower() not in self.NEGATIVE
                and any(
                    pattern.match(confirm_resp.strip())
                    for pattern in self.FINALIZE_PATTERNS
                )
            ):
                state["user_requested_finalize"] = True
                state["needs_followup"] = False
                state["followup_field"] = None
                state["offer_early_termination"] = False

            else:
                state[
                    "offer_early_termination"
                ] = False
                state[
                    "user_confirmed_early_end"
                ] = False

            return state

        # ============================================================
        # GENERATE QUESTION
        # ============================================================

        question = (
            self._generate_conversational_question(
                state=state,
                missing=missing,
                followup_field=followup_field,
                is_first_turn=is_first_turn,
            )
        )

        # ============================================================
        # FALLBACK
        # ============================================================

        validation_question = self._validation_followup_question(target_field, state)
        latest = str(state.get("latest_user_response", ""))
        needs_contextual_down_payment_followup = (
            target_field == "down_payment"
            and not re.search(r"\d", latest)
            and bool(re.search(r"\b(?:hurdle|problem|issue|obstacle)\b", latest, re.IGNORECASE))
        )

        if state.get("clarification_context"):
            question = self._clarifying_question(target_field)
        elif validation_question:
            question = validation_question
        elif needs_contextual_down_payment_followup:
            question = self._fallback_question(target_field, state, is_first_turn)
        elif not question:

            question = self._fallback_question(
                target_field,
                state,
                is_first_turn,
            )

        # ============================================================
        # TRACK EXACT TARGET FIELD
        # ============================================================

        self._track_question(
            state=state,
            field=target_field,
            question=question,
        )

        # ============================================================
        # GREETING
        # ============================================================

        if (
            is_first_turn
            and self.greeting_text
        ):
            if not question.lower().startswith(
                self.greeting_text.lower()
            ):
                question = (
                    f"{self.greeting_text} "
                    f"{question}"
                )

        # ============================================================
        # ASK USER
        # ============================================================

        terminal_ui.print_agent_message(
            question,
            is_first_turn=is_first_turn,
        )

        user_response = (
            terminal_ui
            .get_answer_prompt()
        )

        terminal_ui.print_thinking()

        cleaned = (
            user_response or ""
        ).strip().lower()

        is_stop_command = (
            self._detect_stop_intent(
                user_response
            )
        )

        is_finalize_phrase = self._detect_finalize_intent(
            user_response,
            question,
            target_field,
        )

        is_greeting = (
            self._detect_greeting(
                user_response
            )
        )

        logger.debug(
            "User response=%r cleaned=%r "
            "is_stop=%s is_finalize=%s "
            "is_greeting=%s target_field=%s",
            user_response,
            cleaned,
            is_stop_command,
            is_finalize_phrase,
            is_greeting,
            target_field,
        )

        # ============================================================
        # STOP
        # ============================================================

        if is_stop_command:

            if not history and self.system_prompt:
                history.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )

            history.append(
                {
                    "role": "assistant",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "user",
                    "content": user_response,
                }
            )

            state["conversation_history"] = history
            state["current_question"] = question
            state[
                "latest_user_response"
            ] = user_response

            state["turn_count"] = (
                int(
                    state.get(
                        "turn_count",
                        0,
                    )
                )
                + 1
            )

            state["user_requested_stop"] = True
            state["skip_extraction"] = True
            state["needs_followup"] = False
            state["followup_field"] = None

            state["decision_status"] = (
                "Stopped by User"
            )

            state["decision_summary"] = (
                "User ended the conversation."
            )

            state["lead_step"] = "stop"
            state["summary"] = (
                "User ended the conversation."
            )

            state[
                "qualification_category"
            ] = "stopped"

            state[
                "conversation_status"
            ] = "stopped"

            state[
                "session_tags"
            ] = ["ciap-stopped"]

            state["final_report"] = {
                "status": "Stopped by User",
                "summary": (
                    "User ended the conversation."
                ),
                "stopped_by_user": True,
            }

            return state

        # ============================================================
        # GREETING RESPONSE
        # ============================================================

        if is_greeting:

            if not history and self.system_prompt:
                history.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )

            history.append(
                {
                    "role": "assistant",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "user",
                    "content": user_response,
                }
            )

            state["conversation_history"] = history
            state["current_question"] = question
            state[
                "current_question_field"
            ] = target_field or "general"

            state[
                "latest_user_response"
            ] = user_response

            state["turn_count"] = (
                int(
                    state.get(
                        "turn_count",
                        0,
                    )
                )
                + 1
            )

            state["greeting_detected"] = True
            state["skip_extraction"] = True
            state["needs_followup"] = True
            state[
                "followup_field"
            ] = target_field

            state["lead_step"] = "greeting"
            state["summary"] = (
                "User greeted the assistant."
            )

            state[
                "qualification_category"
            ] = (
                state.get(
                    "qualification_category"
                )
                or "in_progress"
            )

            state[
                "conversation_status"
            ] = "in_progress"

            return state

        if self._detect_clarifying_question(user_response):
            history.append({"role": "assistant", "content": question})
            history.append({"role": "user", "content": user_response})
            state["conversation_history"] = history
            state["current_question"] = question
            state["current_question_field"] = target_field or "general"
            state["latest_user_response"] = user_response
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["skip_extraction"] = True
            state["needs_followup"] = True
            state["followup_field"] = target_field
            state["clarification_context"] = user_response
            return state

        if self._detect_skip_intent(user_response):
            history.append({"role": "assistant", "content": question})
            history.append({"role": "user", "content": user_response})
            skipped_fields = list(state.get("skipped_fields", []))
            if target_field and target_field not in skipped_fields:
                skipped_fields.append(target_field)
            state["skipped_fields"] = skipped_fields
            state["conversation_history"] = history
            state["current_question"] = question
            state["current_question_field"] = target_field or "general"
            state["latest_user_response"] = user_response
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["skip_extraction"] = True
            state["needs_followup"] = True
            state["followup_field"] = None
            state["clarification_context"] = ""
            return state

        # ============================================================
        # USER WANTS TO FINALIZE
        # ============================================================

        if is_finalize_phrase:

            if not history and self.system_prompt:
                history.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )

            history.append(
                {
                    "role": "assistant",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "user",
                    "content": user_response,
                }
            )

            state["conversation_history"] = history
            state["current_question"] = question
            state[
                "current_question_field"
            ] = target_field or "general"

            state[
                "latest_user_response"
            ] = user_response

            state["turn_count"] = (
                int(
                    state.get(
                        "turn_count",
                        0,
                    )
                )
                + 1
            )

            state[
                "user_requested_finalize"
            ] = True

            state["needs_followup"] = False
            state["followup_field"] = None

            state[
                "decision_status"
            ] = "Requires More Info"

            state[
                "decision_summary"
            ] = (
                "User indicated no more "
                "information is available."
            )

            return state

        # ============================================================
        # NORMAL TURN
        # ============================================================

        # A previous greeting or clarification may have skipped extraction.  It
        # applies only to that one turn, never to the next genuine answer.
        state["skip_extraction"] = False
        # LangGraph merges node updates with the existing state; assigning an
        # empty value is reliable across route-backs, while deleting a key is
        # not guaranteed to clear the persisted state.
        state["clarification_context"] = ""

        if not history and self.system_prompt:
            history.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

        history.append(
            {
                "role": "assistant",
                "content": question,
            }
        )

        history.append(
            {
                "role": "user",
                "content": user_response,
            }
        )

        state[
            "current_question"
        ] = question

        state[
            "current_question_field"
        ] = target_field or "general"

        state[
            "latest_user_response"
        ] = user_response

        state[
            "conversation_history"
        ] = history

        state["turn_count"] = (
            int(
                state.get(
                    "turn_count",
                    0,
                )
            )
            + 1
        )

        return state
