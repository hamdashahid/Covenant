from __future__ import annotations

import argparse
from pathlib import Path

from agents.decision_agent import DecisionAgent
from agents.extraction_validation import ExtractionValidationNode
from agents.interview_agent import InterviewAgent
from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA
from core.session_manager import SessionManager
from core import terminal_ui
from graph.ciap_graph import build_ciap_graph
from llm.openai_adapter import OpenAIClientAdapter
from persistence.sqlite_store import SQLiteStore
from rules.rule_evaluator import RuleEvaluator
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


DEFAULT_POLICY = [
    {
        "field": "down_payment",
        "question": "Is a down payment your biggest hurdle right now? If so, roughly how much are you able to put down?",
    },
    {
        "field": "credit_score",
        "question": "What is your credit score?",
    },
    {
        "field": "employment_status",
        "question": "What is your current employment status (employed / self-employed / unemployed)?",
    },
    {
        "field": "employment_years",
        "question": "How many years have you been at your current job or business?",
    },
    {
        "field": "annual_income",
        "question": "What is your annual income (in your local currency)?",
    },
    {
        "field": "total_savings",
        "question": "How much have you saved up so far, in total?",
    },
]
SYSTEM_PROMPT_PATH = Path("config") / "prompts_system_prompt.txt"
MODEL_ID = "gpt-4o"


def _load_system_prompt(path: Path) -> str:
    if not path.exists():
        return "You are the CIAP Interview Agent. Ask one concise question per turn to collect required mortgage eligibility data."
    return path.read_text(encoding="utf-8").strip()


def _load_greeting(path: Path) -> str:
    # Env var takes precedence
    env = os.getenv("CIAP_GREETING")
    if env:
        return env.strip()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="CIAP: Conversational Interview & Assessment Platform")
    parser.add_argument("--session-id", default=None, help="Existing session ID to resume")
    parser.add_argument("--db-path", default="core.db", help="SQLite database path")
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List recent sessions and their persisted conversation tags",
    )
    args = parser.parse_args()

    store = SQLiteStore(db_path=args.db_path)
    if getattr(args, "list_tags", False):
        terminal_ui.print_sessions_by_tag(store.get_sessions_with_tags())
        return

    session_manager = SessionManager(store=store, default_model_id=MODEL_ID)
    session_id, model_id, state = session_manager.start_or_resume(args.session_id, MODEL_ID)

    terminal_ui.print_banner()
    terminal_ui.print_session_info(session_id)

    llm_client = OpenAIClientAdapter(model_id=model_id)
    greeting = _load_greeting(Path("config") / "greeting.txt")
    interview_agent = InterviewAgent(
        DEFAULT_POLICY,
        _load_system_prompt(SYSTEM_PROMPT_PATH),
        llm_client=llm_client,
        greeting_text=greeting,
    )
    extraction_node = ExtractionValidationNode(
        llm_client=llm_client,
        context_builder=ContextBuilder(),
        profile_updater=ProfileUpdater(),
        extraction_schema=EXTRACTION_SCHEMA,
    )
    decision_agent = DecisionAgent(
        rule_evaluator=RuleEvaluator(str(Path("rules") / "eligibility_rules.yaml"))
    )
    # Configure early termination thresholds from environment (defaults if unset)
    try:
        offer = float(os.getenv("EARLY_OFFER_PASS_RATIO", "0.85"))
    except Exception:
        offer = 0.85
    try:
        auto = float(os.getenv("EARLY_AUTO_PASS_RATIO", "1.0"))
    except Exception:
        auto = 1.0
    decision_agent.set_early_termination_thresholds(offer, auto)

    def on_turn_complete(updated_state: dict) -> None:
        store.upsert_profile(
            session_id=session_id,
            profile=updated_state.get("applicant_profile", {}),
            conflicts=updated_state.get("profile_conflicts", []),
        )
        store.replace_messages(session_id=session_id, messages=updated_state.get("conversation_history", []))
        fallback_issues = [
            issue
            for issue in updated_state.get("last_extraction", {}).get("issues", [])
            if "Fallback extraction used:" in str(issue)
        ]
        if fallback_issues:
            terminal_ui.print_error(
                "The service is temporarily unavailable, so I am using a limited local "
                "reader. Please answer with a clear amount or status where possible."
            )
        session_manager.save_state(session_id, updated_state, completed=False)

    def on_completed(updated_state: dict) -> None:
        report = updated_state.get("final_report", {})
        final_tags = []
        if updated_state.get("conversation_tag"):
            final_tags.append(str(updated_state["conversation_tag"]))
        if updated_state.get("session_tags"):
            final_tags.extend([str(tag) for tag in updated_state["session_tags"] if tag not in final_tags])
        if not final_tags and updated_state.get("decision_status") == "Stopped by User":
            final_tags.append("Stopped")
        try:
            store.close_session(session_id=session_id, report=report, tags=final_tags or None)
        except TypeError:
            store.close_session(session_id=session_id, report=report)
        session_manager.save_state(session_id, updated_state, completed=True)

    graph = build_ciap_graph(
        interview_agent=interview_agent,
        extraction_validation_node=extraction_node,
        decision_agent=decision_agent,
        on_turn_complete=on_turn_complete,
        on_completed=on_completed,
    )

    final_state = graph.invoke(state)
    terminal_ui.print_final_result(
        status=final_state.get("decision_status", "Requires More Info"),
        summary=final_state.get("decision_summary", "No summary available"),
        profile=final_state.get("applicant_profile", {}),
        report=final_state.get("final_report", {}),
    )


if __name__ == "__main__":
    main()
