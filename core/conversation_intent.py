from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    ANSWER = "answer"
    SKIP = "skip"
    UNKNOWN = "unknown"
    REFUSAL = "refusal"
    CLARIFICATION = "clarification"
    STOP = "stop"
    GREETING = "greeting"


@dataclass(frozen=True)
class ClassifiedInput:
    intent: Intent
    normalized_text: str


def _normalize(text: str | None) -> str:
    source = (text or "").replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r"[^a-z0-9'/? -]+", " ", source.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip(" .!?\\")


def classify_input(text: str | None) -> ClassifiedInput:
    """Classify conversation control language before field extraction."""
    cleaned = _normalize(text)

    stop_patterns = (
        r"/?(?:stop|end)",
        r"(?:please )?(?:shut ?up|leave me alone)",
        r"(?:please )?stop(?: this| asking| contacting me)?",
        r"end (?:this )?(?:chat|conversation|interview)",
        r"i (?:don'?t|do not) want to (?:continue|answer anymore)",
        r"i (?:don'?t|do not) want to continue (?:this )?(?:chat|conversation|interview)",
        r"(?:don'?t|do not) contact me anymore",
        r"stop contacting me",
        r"i(?:'m| am) not interested anymore",
        r"no longer interested",
    )
    if any(re.fullmatch(pattern, cleaned) for pattern in stop_patterns):
        return ClassifiedInput(Intent.STOP, cleaned)

    skip_patterns = (
        r"skip(?: this)?(?: question)?",
        r"pass",
        r"next(?: question| ques)?",
        r"move (?:to )?(?:the )?next(?: question| ques| quest)?",
        r"ask (?:the )?next(?: question| ques)?",
        r"no[ ,_-]*pass",
    )
    if any(re.fullmatch(pattern, cleaned) for pattern in skip_patterns):
        return ClassifiedInput(Intent.SKIP, cleaned)

    refusal_patterns = (
        r"i (?:don'?t|do not) want to (?:tell|say|share|answer)(?: (?:it|you|u|that))?",
        r"(?:i'd|i would) rather not(?: say| tell| share| answer)?",
        r"none of your business",
        r"prefer not to (?:say|tell|share|answer)",
    )
    if any(re.fullmatch(pattern, cleaned) for pattern in refusal_patterns):
        return ClassifiedInput(Intent.REFUSAL, cleaned)

    unknown_patterns = (
        r"(?:i )?(?:don'?t|do not) know",
        r"(?:i )?(?:don'?t|do not|can'?t|cannot) remember",
        r"(?:i have )?no idea",
        r"not (?:sure|certain|remember|pretty sure|preety sure)",
        r"unknown",
        r"no answer",
        r"(?:i )?(?:couldn'?t|could not|can'?t|cannot) estimate(?: it| that| my score)?(?: right now)?",
    )
    if any(re.fullmatch(pattern, cleaned) for pattern in unknown_patterns):
        return ClassifiedInput(Intent.UNKNOWN, cleaned)

    clarification_patterns = (
        r"(?:what|which) (?:do you mean|range|amount|one).*",
        r"(?:like )?what",
        r"how (?:much|many|do you mean).*",
        r"(?:can|could) you (?:explain|clarify).*",
        r"in which range.*",
        r"what (?:kind|sort).*",
        r"i(?:'m| am) not sure what you mean.*",
        r"what (?:is|are) (?:a |the )?(?:debt payments?|credit score|down payment|annual income|savings?|employment status)",
    )
    if any(re.fullmatch(pattern, cleaned) for pattern in clarification_patterns):
        return ClassifiedInput(Intent.CLARIFICATION, cleaned)

    greetings = ("hi", "hello", "hey", "hi there", "hello there", "hey there")
    if cleaned in greetings or re.fullmatch(r"(?:hi|hello|hey) [a-z]+", cleaned):
        return ClassifiedInput(Intent.GREETING, cleaned)

    return ClassifiedInput(Intent.ANSWER, cleaned)
