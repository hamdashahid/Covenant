from __future__ import annotations

import json
import logging
import math
import re
from datetime import date
from typing import Any

from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater

logger = logging.getLogger(__name__)


class ExtractionValidationNode:
    def __init__(
        self,
        llm_client: Any,
        context_builder: ContextBuilder,
        profile_updater: ProfileUpdater,
        extraction_schema: dict[str, str],
    ) -> None:
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.profile_updater = profile_updater
        self.extraction_schema = extraction_schema

    def _normalize_numeric_text(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        # Remove currency symbols and comma grouping before parsing.
        text = re.sub(r"[\$₹£€]", "", text)
        text = text.replace(",", "")

        # Extract the first numeric token, allowing modifiers like k / m and
        # fuzzy prefixes such as above, below, around, or plus signs.
        match = re.search(r"([-+]?\d+(?:\.\d+)?)([kKmM]?)", text)
        if match:
            number_text = match.group(1)
            suffix = match.group(2).lower()
            try:
                value_float = float(number_text)
            except (TypeError, ValueError, OverflowError):
                return text
            if suffix == "k":
                value_float *= 1_000
            elif suffix == "m":
                value_float *= 1_000_000
            return str(value_float)

        word_number = self._parse_number_words(text)
        if word_number is not None:
            return str(word_number)

        # Preserve the cleaned string if no numeric token was found.
        return text

    def _parse_number_words(self, text: str) -> float | None:
        """Parse common spoken amounts such as 'ninety thousand'."""
        cleaned = re.sub(r"[^a-z -]", " ", text.lower().replace("’", "'")).replace("-", " ")
        tokens = [
            token for token in cleaned.split()
            if token not in {
                "and", "about", "around", "roughly", "close", "to", "annually",
                "yearly", "per", "year", "i", "my", "income", "earn", "make",
                "salary", "is", "approximately", "almost",
            }
        ]
        small = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }
        if not tokens or any(token not in small and token not in {"hundred", "thousand", "million", "billion"} for token in tokens):
            return None
        total = 0
        current = 0
        for token in tokens:
            if token in small:
                current += small[token]
            elif token == "hundred":
                current = max(current, 1) * 100
            elif token in {"thousand", "million", "billion"}:
                scale = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}[token]
                total += max(current, 1) * scale
                current = 0
        return float(total + current)

    def _parse_float(self, raw: Any) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        text = self._normalize_numeric_text(raw)
        if not text:
            return None
        try:
            return float(text)
        except (TypeError, ValueError, OverflowError):
            return None

    def _parse_int(self, raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        text = self._normalize_numeric_text(raw)
        if not text:
            return None
        try:
            if "." in text:
                numeric = float(text)
                if numeric.is_integer():
                    return int(numeric)
                return None
            return int(text)
        except (TypeError, ValueError, OverflowError):
            return None

    def _normalize_employment_status(self, value: Any) -> str | None:
        """Map explicit, everyday employment descriptions to rule values."""
        text = str(value or "").strip().lower()
        if not text:
            return None
        if re.fullmatch(r"self(?:[ -]+emplo\w*)?|business[ -]?man", text):
            return "self-employed"
        if re.search(r"\b(self[ -]?employ\w*|freelanc\w*|contractor|business[ -]?man|own (?:a )?business|business owner)\b", text):
            return "self-employed"
        if re.search(
            r"\b(retir\w*|pension(?:er|ed)?|not working anymore|no longer working|"
            r"unemploy\w*|out of work|no job|jobless|laid[ -]?off|layoff|between jobs|"
            r"looking for (?:a )?(?:job|work)|seeking (?:a )?(?:job|work)|"
            r"lost my job|terminated|dismissed|made redundant)\b",
            text,
        ):
            return "unemployed"
        if re.fullmatch(r"emp|emplo\w*", text):
            return "employed"
        if re.search(r"\b(emp|employ\w*|full[ -]?time|part[ -]?time|working)\b", text):
            return "employed"
        return None

    def _parse_non_negative_amount(self, raw: Any) -> float | None:
        """Understand everyday zero answers without treating uncertainty as zero."""
        text = re.sub(r"\s+", " ", str(raw or "").replace("’", "'").strip().lower()).strip(".!?")
        zero_answers = {
            "no",
            "none",
            "nothing",
            "zero",
            "no debt",
            "no debts",
            "no savings",
            "no down payment",
            "no amount",
            "i have no amount",
            "i've got no amount",
            "no amount available",
            "not any",
        }
        if text in zero_answers:
            return 0.0
        zero_patterns = (
            r"(?:i (?:have|got) )?no (?:down )?payments?",
            r"i (?:don'?t|do not) pay (?:any )?(?:debt|debts|loans?)",
            r"(?:i have )?no (?:monthly )?(?:debt|debts|loans?|repayments?)",
            r"nothing (?:to pay|toward debt|for the down payment)",
            r"(?:i have )?nothing available for (?:a |the )?down payment",
            r"i (?:don'?t|do not) make any debt payments?",
        )
        if any(re.fullmatch(pattern, text) for pattern in zero_patterns):
            return 0.0

        # Meaning-based zero detection. Allow conversational filler and words
        # such as "toward", while requiring both a clear negation and a debt,
        # savings, or down-payment action. This avoids an endless phrase list.
        debt_zero = bool(
            re.search(r"\b(?:don'?t|do not|never|no)\b.*\b(?:pay|owe|have|make)\b.*\b(?:debts?|loans?|repayments?)\b", text)
            or re.search(r"\b(?:have|owe|make)\b.*\bno\b.*\b(?:debts?|loans?|repayments?)\b", text)
        )
        down_payment_zero = bool(
            re.search(r"\bno amount\b", text)
            or re.search(r"\b(?:no|nothing|zero)\b.*\b(?:down payment|payment|put down|upfront)\b", text)
            or re.search(r"\b(?:can'?t|cannot|don'?t|do not)\b.*\b(?:put|pay)\b.*\b(?:down|upfront)\b", text)
        )
        savings_zero = bool(
            re.search(r"\b(?:no|nothing|zero)\b.*\b(?:savings?|saved)\b", text)
            or re.search(r"\b(?:haven'?t|have not|don'?t|do not)\b.*\b(?:save|saved)\b", text)
        )
        if debt_zero or down_payment_zero or savings_zero:
            return 0.0
        return self._parse_float(raw)

    def _infer_fields_from_question(
        self, raw: str, latest_question: str, current_field: str | None = None
    ) -> dict[str, Any]:
        response = str(raw or "").strip()
        question = str(latest_question or "").lower()
        inferred: dict[str, Any] = {}

        if not response:
            return inferred

        def is_target(field: str, question_pattern: str) -> bool:
            # Once the interview agent identifies the field being asked, that
            # state is authoritative. Acknowledgements in the question may
            # mention older values and must never make them extraction targets.
            if current_field:
                return current_field == field
            return bool(re.search(question_pattern, question))

        if is_target("annual_income", r"annual income|income"):
            value = self._parse_float(response)
            if value is not None:
                inferred["annual_income"] = value

        if is_target("monthly_debt", r"monthly debt|debt payment|debt"):
            value = self._parse_non_negative_amount(response)
            if value is not None:
                inferred["monthly_debt"] = value

        if is_target("credit_score", r"credit score|score"):
            value = self._parse_int(response)
            if value is not None:
                inferred["credit_score"] = value

        if is_target("employment_status", r"employment status"):
            normalized = self._normalize_employment_status(response)
            if normalized:
                inferred["employment_status"] = normalized

        if is_target("employment_years", r"years|how long have you been|current job|business") and not re.search(r"\d\s*/\s*\d", response):
            value = self._parse_float(response)
            if value is not None:
                inferred["employment_years"] = value

        if is_target("property_value", r"property value|value of the property|property you want|price of the property"):
            value = self._parse_float(response)
            if value is not None:
                inferred["property_value"] = value

        if is_target("requested_loan_amount", r"loan amount|how much loan|requesting|loan you're requesting|loan you want"):
            value = self._parse_float(response)
            if value is not None:
                inferred["requested_loan_amount"] = value

        if is_target("down_payment", r"down payment|pay upfront"):
            value = self._parse_non_negative_amount(response)
            if value is not None:
                inferred["down_payment"] = value

        if is_target("total_savings", r"saved up|saved so far|how much have you saved|total savings"):
            value = self._parse_non_negative_amount(response)
            if value is not None:
                inferred["total_savings"] = value

        return inferred

    def _limit_fields_to_latest_answer(
        self,
        fields: dict[str, Any],
        response: str,
        current_field: str | None,
    ) -> dict[str, Any]:
        """Prevent a model from copying unrelated values out of prior context."""
        if current_field not in self.extraction_schema:
            return fields

        text = str(response or "").lower()
        explicit_patterns = {
            "annual_income": r"\b(?:annual income|income|earn|salary)\b",
            "monthly_debt": r"\b(?:monthly debt|debt|repayment)\b",
            "credit_score": r"\b(?:credit|score)\b",
            "employment_status": r"\b(?:employed|self[ -]?employed|unemployed|retired|working|jobless)\b",
            "employment_years": r"\b(?:years?|months?|job|business)\b",
            "property_value": r"\b(?:property|house|home|purchase price)\b",
            "requested_loan_amount": r"\b(?:loan|borrow)\b",
            "down_payment": r"\b(?:down payment|put down|upfront)\b",
            "total_savings": r"\b(?:savings?|saved)\b",
        }
        allowed = {current_field}
        allowed.update(
            field
            for field, pattern in explicit_patterns.items()
            if re.search(pattern, text)
        )
        return {field: value for field, value in fields.items() if field in allowed}

    def _coerce_and_validate(self, fields: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        validated: dict[str, Any] = {}
        issues: list[str] = []

        if "annual_income" in fields:
            value = self._parse_float(fields["annual_income"])
            if value is not None and math.isfinite(value) and 0 < value <= 1_000_000_000_000:
                validated["annual_income"] = value
            else:
                issues.append("annual_income must be a realistic positive yearly amount")

        if "monthly_debt" in fields:
            value = self._parse_float(fields["monthly_debt"])
            if value is not None and math.isfinite(value) and 0 <= value <= 1_000_000_000_000:
                validated["monthly_debt"] = value
            else:
                issues.append("monthly_debt must be >= 0")

        if "credit_score" in fields:
            value = self._parse_int(fields["credit_score"])
            if value is not None and 300 <= value <= 850:
                validated["credit_score"] = value
            else:
                issues.append("credit_score must be between 300 and 850")

        if "employment_status" in fields:
            normalized = self._normalize_employment_status(fields["employment_status"])
            if normalized:
                validated["employment_status"] = normalized
            else:
                issues.append("employment_status is invalid")

        if "requested_loan_amount" in fields:
            value = self._parse_float(fields["requested_loan_amount"])
            if value is not None and value > 0:
                validated["requested_loan_amount"] = value
            else:
                issues.append("requested_loan_amount must be > 0")

        if "employment_years" in fields:
            value = self._parse_float(fields["employment_years"])
            if value is not None and math.isfinite(value) and 0 <= value <= 80:
                validated["employment_years"] = value
            else:
                issues.append("employment_years must be between 0 and 80")

        if "property_value" in fields:
            value = self._parse_float(fields["property_value"])
            if value is not None and value > 0:
                validated["property_value"] = value
            else:
                issues.append("property_value must be > 0")

        if "down_payment" in fields:
            value = self._parse_float(fields["down_payment"])
            if value is not None and math.isfinite(value) and 0 <= value <= 1_000_000_000_000_000:
                validated["down_payment"] = value
            else:
                issues.append("down_payment must be >= 0")

        if "total_savings" in fields:
            value = self._parse_float(fields["total_savings"])
            if value is not None and math.isfinite(value) and 0 <= value <= 1_000_000_000_000_000:
                validated["total_savings"] = value
            else:
                issues.append("total_savings must be >= 0")

        return validated, issues

    def _parse_response(self, raw: str) -> tuple[dict[str, Any], float, list[str]]:
        try:
            parsed = json.loads(raw)
            fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
            confidence = float(parsed.get("confidence", 0.0)) if isinstance(parsed, dict) else 0.0
            issues = parsed.get("issues", []) if isinstance(parsed, dict) else ["Invalid extractor response shape"]
            if not isinstance(fields, dict):
                fields = {}
                issues.append("fields must be an object")
            if not isinstance(issues, list):
                issues = ["issues must be a list"]
            return fields, confidence, [str(i) for i in issues]
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}, 0.0, ["Extractor response was not valid JSON"]

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self.context_builder.build_extraction_prompt(
            conversation_history=state.get("conversation_history", []),
            profile=state.get("applicant_profile", {}),
            latest_question=state.get("current_question", ""),
            latest_user_response=state.get("latest_user_response", ""),
            schema=self.extraction_schema,
        )
        interpreted = state.get("interpreted_input", {}) or {}
        interpreted_value = interpreted.get("value") if isinstance(interpreted, dict) else None
        interpreted_current_value = None
        if (
            isinstance(interpreted, dict)
            and interpreted.get("intent") == "answer"
            and interpreted_value is not None
            and state.get("current_question_field")
        ):
            interpreted_current_value = interpreted_value
        try:
            raw = self.llm_client.extract_structured(prompt, state.get("latest_user_response", ""))
            extraction_failed = False
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("LLM extraction failed", exc_info=True)
            raw = json.dumps(
                {
                    "fields": {},
                    "confidence": 0.0,
                    "issues": ["LLM extraction failed; deterministic fallback unavailable"],
                }
            )
            extraction_failed = True
        fields, confidence, parse_issues = self._parse_response(raw)
        current_field = state.get("current_question_field")
        if interpreted_current_value is not None and current_field:
            # Preserve the field-locked semantic interpretation while still
            # allowing other explicitly volunteered facts from the same reply.
            fields[current_field] = interpreted_current_value
            confidence = max(confidence, float(interpreted.get("confidence", 0.0)))
        fields = self._limit_fields_to_latest_answer(
            fields,
            state.get("latest_user_response", ""),
            current_field,
        )

        latest_response = str(state.get("latest_user_response", ""))

        # A literal negative answer is authoritative.  Do not allow an LLM or
        # intent normalizer to silently turn "-1" into zero, because that can
        # make invalid financial data look valid and advance the interview.
        non_negative_numeric_fields = {
            "down_payment",
            "monthly_debt",
            "total_savings",
            "employment_years",
            "annual_income",
            "property_value",
            "requested_loan_amount",
        }
        negative_match = re.search(r"(?<!\w)-\s*\d+(?:[.,]\d+)?(?:\s*[kKmM])?", latest_response)
        if current_field in non_negative_numeric_fields and negative_match:
            negative_value = self._parse_float(negative_match.group(0).replace(" ", ""))
            if negative_value is not None:
                fields[current_field] = negative_value

        # Normalize time periods explicitly stated by the applicant instead of
        # trusting the model to guess whether a number is monthly or yearly.
        stated_number = self._parse_float(latest_response)
        lowered_response = latest_response.lower()
        if stated_number is not None and current_field == "annual_income":
            if re.search(r"\b(?:per\s+month|monthly|a\s+month|each\s+month)\b", lowered_response):
                fields["annual_income"] = stated_number * 12
            elif re.search(r"\b(?:per\s+week|weekly|a\s+week|each\s+week)\b", lowered_response):
                fields["annual_income"] = stated_number * 52
        elif stated_number is not None and current_field == "monthly_debt":
            if re.search(r"\b(?:per\s+year|annually|annual|a\s+year|each\s+year)\b", lowered_response):
                fields["monthly_debt"] = stated_number / 12
        elif stated_number is not None and current_field == "employment_years":
            if re.search(r"\bmonths?\b", lowered_response):
                fields["employment_years"] = stated_number / 12
            elif re.search(r"\bweeks?\b", lowered_response):
                fields["employment_years"] = stated_number / 52

        if current_field == "employment_years":
            start_year_match = re.search(
                r"\b(?:(?:from|since|at|in)\s+|(?:started|began)(?:\s+my\s+business)?(?:\s+in|\s+at|\s+from)?\s*)(19\d{2}|20\d{2})\b",
                latest_response,
                re.IGNORECASE,
            )
            if start_year_match:
                start_year = int(start_year_match.group(1))
                if start_year <= date.today().year:
                    fields["employment_years"] = float(date.today().year - start_year)

        if current_field == "employment_years" and re.search(
            r"\d\s*/\s*\d",
            latest_response,
        ):
            fields.pop("employment_years", None)
            parse_issues.append("employment_years is ambiguous")

        validated_fields, validation_issues = self._coerce_and_validate(fields)

        # The deterministic field-specific reader is especially important in
        # degraded mode: the provider fallback only sees the answer text and
        # cannot always know which question is currently being answered.
        if confidence < 0.30 or not validated_fields:
            inferred_fields = self._infer_fields_from_question(
                state.get("latest_user_response", ""),
                state.get("current_question", ""),
                state.get("current_question_field"),
            )
            inferred_validated, inferred_issues = self._coerce_and_validate(inferred_fields)
            if inferred_validated and not validated_fields:
                validated_fields = inferred_validated
                validation_issues = inferred_issues
            elif not confidence < 0.30:
                validation_issues.extend(inferred_issues)

        # Apply the same scope guard after deterministic inference. This is a
        # final defense against an acknowledgement about an earlier field
        # causing that field to be overwritten by the latest numeric answer.
        validated_fields = self._limit_fields_to_latest_answer(
            validated_fields,
            state.get("latest_user_response", ""),
            current_field,
        )

        if current_field == "employment_status" and "employment_status" not in validated_fields:
            validation_issues.append("employment_status is invalid")
        if current_field == "monthly_debt" and "monthly_debt" not in validated_fields:
            validation_issues.append("monthly_debt is unclear")
        if current_field and current_field not in validated_fields and not any(
            str(issue).startswith(current_field) for issue in validation_issues
        ):
            validation_issues.append(f"{current_field} is unclear")

        if confidence < 0.30:
            validation_issues.append("Extraction confidence too low")

        existing_profile = state.get("applicant_profile", {})
        corrected_fields = [
            field
            for field, value in validated_fields.items()
            if field in existing_profile and existing_profile[field] != value
        ]
        merged, conflicts = self.profile_updater.merge(
            existing_profile,
            validated_fields,
        )

        state["applicant_profile"] = merged
        answered_fields = list(state.get("answered_fields", []))
        for field in validated_fields:
            if field not in answered_fields:
                answered_fields.append(field)
        state["answered_fields"] = answered_fields
        state["profile_conflicts"] = list(state.get("profile_conflicts", [])) + conflicts
        state["recent_profile_corrections"] = corrected_fields
        state["last_extraction"] = {
            "raw": raw,
            "fields": fields,
            "validated_fields": validated_fields,
            "confidence": confidence,
            "issues": parse_issues + validation_issues,
        }
        return state
