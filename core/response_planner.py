from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HumanResponsePlan:
    mode: str
    instruction: str
    allow_value_echo: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_response_plan(
    state: dict[str, Any],
    target_field: str | None,
    *,
    is_opening: bool = False,
    is_followup: bool = False,
) -> HumanResponsePlan:
    """Choose the conversational function before generating any wording."""
    if is_opening:
        return HumanResponsePlan(
            "opening",
            "Introduce yourself briefly and ask the first-home question without praise or extra questions.",
        )

    corrections = list(state.get("recent_profile_corrections", []))
    if corrections:
        return HumanResponsePlan(
            "correction",
            "Ask the next question naturally without repeating the corrected value. "
            "The application will add one brief correction confirmation; do not add another or praise it.",
        )

    latest = str(state.get("latest_user_response", "")).lower()
    sensitive = re.search(
        r"\b(?:laid[ -]?off|lost my job|unemploy\w*|between jobs|fired|terminated|"
        r"made redundant|struggling|difficult|hard for me|no savings|nothing saved)\b",
        latest,
    )
    if sensitive:
        return HumanResponsePlan(
            "empathetic_transition",
            "Use one calm, sincere acknowledgement of the situation, then ask the next necessary question. "
            "Do not repeat the applicant's wording, value, or employment label.",
        )

    if is_followup or state.get("clarification_context"):
        return HumanResponsePlan(
            "clarification",
            "Explain what is missing in plain language and ask one easier follow-up. Do not say the answer was noted.",
            allow_value_echo=True,
        )

    turn_count = int(state.get("turn_count", 0))
    if turn_count % 3 == 1:
        return HumanResponsePlan(
            "brief_transition",
            "Ask one short neutral question. Do not add an acknowledgement, praise, or a sentence before it.",
        )
    if turn_count % 3 == 2:
        return HumanResponsePlan(
            "contextual_transition",
            "Put useful context inside the question itself, such as asking how long they have been freelancing. "
            "Do not add praise, evaluation, or a separate acknowledgement sentence.",
        )
    return HumanResponsePlan(
        "direct_transition",
        "Ask the next question directly. The application may add a brief value-aware reaction on selected turns; "
        "do not add a generic acknowledgement yourself.",
    )
