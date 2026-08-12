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

def print_final_result(status: str, summary: str, profile: dict[str, Any], report: dict[str, Any] | None = None) -> None:
    width = _width()

    if status == "Eligible":
        color = Fore.GREEN
        headline = "GOOD NEWS — YOU LOOK ELIGIBLE"
        icon = "✅"
    elif status == "Ineligible":
        color = Fore.RED
        headline = "NOT ELIGIBLE YET"
        icon = "❌"
    elif status == "Stopped by User":
        color = Fore.MAGENTA
        headline = "CONVERSATION ENDED BY USER"
        icon = "🛑"
    else:
        color = Fore.YELLOW
        headline = "MORE INFORMATION NEEDED"
        icon = "⚠️"

    # ---- Missing-info case: no rule breakdown available ----
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

    # ---- Plain-English opening paragraph ----
    passed_count = sum(1 for r in report["rule_breakdown"] if r["passed"])
    total_count = len(report["rule_breakdown"])
    print()
    if status == "Eligible":
        print(
            Style.BRIGHT
            + f"Your application passed all {total_count} eligibility checks. "
            + "Based on the numbers you provided, you meet the lender's requirements "
            + "for income, debt, credit, job stability, and the loan itself."
            + Style.RESET_ALL
        )
    else:
        print(
            Style.BRIGHT
            + f"Your application passed {passed_count} out of {total_count} eligibility checks. "
            + "A few areas below need improvement before you'd typically qualify — "
            + "see exactly which ones and why."
            + Style.RESET_ALL
        )
        print(
            Style.BRIGHT
            + "These are commonly fixable with time or a stronger profile, so a few focused "
            + "changes could make a real difference."
            + Style.RESET_ALL
        )

    # ---- Rule-by-rule breakdown ----
    print()
    print(Fore.BLUE + Style.BRIGHT + "-" * width)
    print(Fore.BLUE + Style.BRIGHT + "Rule-by-Rule Breakdown".center(width))
    print(Fore.BLUE + Style.BRIGHT + "-" * width + Style.RESET_ALL)

    metrics = report.get("metrics", {})
    bar_map = {
        "Debt-to-Income Ratio": (metrics.get("dti_ratio", 0), 0.43),
        "Loan-to-Value Ratio": (metrics.get("ltv_ratio", 0), 0.95),
    }

    for rule in report["rule_breakdown"]:
        mark = (Fore.GREEN + Style.BRIGHT + "✔ PASS" + Style.RESET_ALL) if rule["passed"] else (
            Fore.RED + Style.BRIGHT + "✘ FAIL" + Style.RESET_ALL
        )
        print()
        print(f"  {mark}  " + Style.BRIGHT + rule["name"] + Style.RESET_ALL)
        print(f"        Your value : {Fore.YELLOW}{rule['value_display']}{Style.RESET_ALL}")
        print(f"        Requirement: {Style.DIM}{rule['threshold_display']}{Style.RESET_ALL}")
        if rule["name"] in bar_map:
            value, threshold = bar_map[rule["name"]]
            print(f"        {_bar(value / threshold if threshold else 0, good=rule['passed'])}")
        # word-wrap the explanation to keep it readable
        explanation = rule["explanation"]
        indent = "        "
        line = indent
        for word in explanation.split():
            if len(line) + len(word) + 1 > width:
                print(line)
                line = indent + word
            else:
                line += (" " if line != indent else "") + word
        print(line)

    print()
    print(Fore.BLUE + Style.BRIGHT + "-" * width + Style.RESET_ALL)

    # ---- Final verdict box ----
    print()
    print(color + Style.BRIGHT + "=" * width)
    print(color + Style.BRIGHT + f"FINAL VERDICT: {status.upper()}".center(width))
    print(color + Style.BRIGHT + "=" * width + Style.RESET_ALL)

    if status != "Eligible":
        failed = [r["name"] for r in report["rule_breakdown"] if not r["passed"]]
        print()
        print(Style.BRIGHT + "What would help most:" + Style.RESET_ALL)
        for name in failed:
            print(f"  • Improve: {Fore.YELLOW}{name}{Style.RESET_ALL}")

    print_summary(profile)