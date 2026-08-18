from __future__ import annotations

import shutil
from typing import Any

from colorama import Fore, Style
from colorama import init as colorama_init

from core.schemas import FIELD_LABELS

colorama_init(autoreset=True)


def _width(default: int = 60) -> int:
    try:
        return max(50, min(shutil.get_terminal_size().columns, 92))
    except Exception:
        return default


def _bar(fraction: float, length: int = 24, good: bool = True) -> str:
    """Render a small [####------] style progress bar for a ratio (0.0 - 1.0+)."""
    fraction = max(0.0, min(fraction, 1.0))
    filled = int(round(fraction * length))
    empty = length - filled
    color = Fore.GREEN if good else Fore.RED
    return color + ("█" * filled) + Style.DIM + ("░" * empty) + Style.RESET_ALL


# ---------------------------------------------------------------------------
# Intro / session
# ---------------------------------------------------------------------------

def print_banner() -> None:
    width = _width()
    title = "CIAP — Mortgage Eligibility Assistant"
    print(Fore.CYAN + Style.BRIGHT + "=" * width)
    print(Fore.CYAN + Style.BRIGHT + title.center(width))
    print(Fore.CYAN + Style.BRIGHT + "=" * width + Style.RESET_ALL)


def print_session_info(session_id: str) -> None:
    print(Fore.YELLOW + f"Session ID: {session_id}" + Style.RESET_ALL)
    print(
        Fore.WHITE
        + Style.DIM
        + "Ctrl+C to exit anytime. Resume later with: "
        + f"python main.py --session-id {session_id}"
        + Style.RESET_ALL
    )


def print_sessions_by_tag(sessions: list[dict[str, Any]]) -> None:
    width = _width()
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * width)
    print(Fore.CYAN + Style.BRIGHT + "Conversations by Tag".center(width))
    print(Fore.CYAN + Style.BRIGHT + "=" * width + Style.RESET_ALL)
    if not sessions:
        print(Fore.YELLOW + "No sessions were found." + Style.RESET_ALL)
        return
    for session in sessions:
        tags = session.get("tags") or []
        tag_text = ", ".join(tags) if tags else "none"
        print(
            Fore.WHITE
            + f"{session['session_id']} [{session['session_state']}] "
            + Fore.YELLOW
            + f"tags: {tag_text}"
            + Style.RESET_ALL
        )
    print(Fore.CYAN + Style.BRIGHT + "=" * width + Style.RESET_ALL)


# ---------------------------------------------------------------------------
# Interview turn
# ---------------------------------------------------------------------------

def print_agent_message(message: str, is_first_turn: bool = False) -> None:
    print()
    label = Fore.GREEN + Style.BRIGHT + "CIAP: " + Style.RESET_ALL
    print(label + message)


def get_answer_prompt() -> str:
    try:
        return input(Fore.CYAN + Style.BRIGHT + "You: " + Style.RESET_ALL).strip()
    except (EOFError, OSError, RuntimeError):
        return ""


def print_error(message: str) -> None:
    print(Fore.RED + Style.BRIGHT + f"[!] {message}" + Style.RESET_ALL)


def print_thinking() -> None:
    print(Fore.WHITE + Style.DIM + "  ...processing your answer" + Style.RESET_ALL)


# ---------------------------------------------------------------------------
# Collected profile summary (shown mid-flow, e.g. missing-info cases)
# ---------------------------------------------------------------------------

def print_summary(profile: dict[str, Any]) -> None:
    width = _width()
    print()
    print(Fore.BLUE + Style.BRIGHT + "-" * width)
    print(Fore.BLUE + Style.BRIGHT + "Collected Information".center(width))
    print(Fore.BLUE + Style.BRIGHT + "-" * width + Style.RESET_ALL)
    for key, label in FIELD_LABELS.items():
        value = profile.get(key, "—")
        print(f"  {label:<24}: " + Fore.YELLOW + str(value) + Style.RESET_ALL)
    print(Fore.BLUE + Style.BRIGHT + "-" * width + Style.RESET_ALL)


# ---------------------------------------------------------------------------
# Final decision — the big, layman-friendly, visually rich report
# ---------------------------------------------------------------------------

def print_tag_view(store: Any) -> None:
    width = _width()
    tags = store.get_available_tags()
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * width)
    print(Fore.CYAN + Style.BRIGHT + "Conversations by Tag".center(width))
    print(Fore.CYAN + Style.BRIGHT + "=" * width + Style.RESET_ALL)
    if not tags:
        print(Fore.YELLOW + "No final tags have been assigned yet." + Style.RESET_ALL)
        return
    for tag in tags:
        print()
        print(Fore.MAGENTA + Style.BRIGHT + tag + Style.RESET_ALL)
        conversations = store.get_conversations_by_tag(tag)
        if not conversations:
            print("  (none)")
            continue
        for conversation in conversations:
            status = conversation.get("closed_at") and "closed" or "open"
            print(f"  - {conversation['session_id']} [{status}]")


def _conversational_message(status: str, report: dict[str, Any] | None) -> tuple[str, str | None]:
    """Build a short, human, Sir's-bot-style message instead of a rule-by-rule audit.

    Returns (main_message, follow_up_question_or_None).
    """
    rule_breakdown = (report or {}).get("rule_breakdown", [])
    failed = [r["name"] for r in rule_breakdown if not r.get("passed")]

    if status == "Eligible":
        return (
            "Great news! Based on everything you've shared, you check all our "
            "boxes — your income, credit, and job stability all look solid for "
            "moving forward.",
            None,
        )

    if status == "Ineligible":
        failed_set = set(failed)
        if "Annual Income" in failed_set and "Credit Score" in failed_set:
            reason = "your income and credit score are the two areas holding you back right now"
        elif "Credit Score" in failed_set:
            reason = "your credit score is the main thing holding you back right now"
        elif "Annual Income" in failed_set:
            reason = "your income is a bit below what's typically needed"
        elif "Debt-to-Income Ratio" in failed_set:
            reason = "your monthly debt relative to your income is a bit higher than lenders typically allow"
        elif "Employment Status" in failed_set or "Job Stability" in failed_set:
            reason = "your employment history is the main gap right now"
        elif "Loan-to-Value Ratio" in failed_set:
            reason = "the loan amount relative to the property's value is a bit high"
        elif "Down Payment" in failed_set:
            reason = "your down payment is a bit below what's typically required"
        else:
            reason = "a couple of areas in your profile need a bit more work"

        return (
            f"I've gone through your details, and {reason} — but that's usually "
            "fixable with a bit of time or a stronger profile.",
            "Would you like a few tips on what could help before we revisit this together?",
        )

    return (None, None)


def print_final_result(status: str, summary: str, profile: dict[str, Any], report: dict[str, Any] | None = None) -> None:
    width = _width()

    if status == "Eligible":
        color = Fore.GREEN
        headline = "GOOD NEWS — YOU ARE ELIGIBLE"
        icon = "\u2705"
    elif status == "Ineligible":
        color = Fore.RED
        headline = "NOT ELIGIBLE YET"
        icon = "\u274c"
    elif status == "Stopped by User":
        color = Fore.MAGENTA
        headline = "CONVERSATION ENDED BY USER"
        icon = "\U0001f6d1"
    else:
        color = Fore.YELLOW
        headline = "MORE INFORMATION NEEDED"
        icon = "\u26a0\ufe0f"

    # ---- Missing-info / stopped case: no rule breakdown available ----
    if not report or "rule_breakdown" not in report:
        print()
        print(color + Style.BRIGHT + "=" * width)
        print(color + Style.BRIGHT + f"{icon}  {headline}".center(width))
        print(color + Style.BRIGHT + "=" * width + Style.RESET_ALL)
        print(Style.BRIGHT + summary + Style.RESET_ALL)
        print(color + "=" * width + Style.RESET_ALL)
        print_summary(profile)
        return

    # ---- Header banner ----
    print()
    print(color + Style.BRIGHT + "=" * width)
    print(color + Style.BRIGHT + f"{icon}  {headline}".center(width))
    print(color + Style.BRIGHT + "=" * width + Style.RESET_ALL)

    # ---- Conversational message + call-to-action (Sir's-bot style) ----
    message, follow_up = _conversational_message(status, report)
    print()
    if message:
        print(Style.BRIGHT + message + Style.RESET_ALL)
    if follow_up:
        print()
        print(follow_up)

    print()
    print(color + "=" * width + Style.RESET_ALL)

    print_summary(profile)