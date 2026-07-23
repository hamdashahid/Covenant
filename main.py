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
from graph.ciap_graph import build_ciap_graph
from llm.claude_adapter import ClaudeClientAdapter
from persistence.sqlite_store import SQLiteStore
from rules.rule_evaluator import RuleEvaluator

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


DEFAULT_POLICY = [
    {
        "field": "annual_income",
        "question": "What is your annual income (in your local currency)?",
    },
    {
        "field": "monthly_debt",
        "question": "What is your total monthly debt payment?",
    },
    {
        "field": "credit_score",
        "question": "What is your credit score?",
    },
    {
        "field": "employment_status",
        "question": "What is your current employment status?",
    },
]
SYSTEM_PROMPT_PATH = Path("config") / "prompts_system_prompt.txt"
MODEL_ID = "claude-sonnet-4-6"


def _load_system_prompt(path: Path) -> str:
    if not path.exists():
        return "You are the CIAP Interview Agent. Ask one concise question per turn to collect required mortgage eligibility data."
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="CIAP: Conversational Interview & Assessment Platform")
    parser.add_argument("--session-id", default=None, help="Existing session ID to resume")
    parser.add_argument("--db-path", default="core.db", help="SQLite database path")
    args = parser.parse_args()

    store = SQLiteStore(db_path=args.db_path)
    session_manager = SessionManager(store=store, default_model_id=MODEL_ID)
    session_id, model_id, state = session_manager.start_or_resume(args.session_id, MODEL_ID)

    print(f"Session ID: {session_id}")
    print("Starting CIAP interview... (Ctrl+C to exit; use --session-id to resume)")

    llm_client = ClaudeClientAdapter(model_id=model_id)
    interview_agent = InterviewAgent(DEFAULT_POLICY, _load_system_prompt(SYSTEM_PROMPT_PATH))
    extraction_node = ExtractionValidationNode(
        llm_client=llm_client,
        context_builder=ContextBuilder(),
        profile_updater=ProfileUpdater(),
        extraction_schema=EXTRACTION_SCHEMA,
    )
    decision_agent = DecisionAgent(
        rule_evaluator=RuleEvaluator(str(Path("rules") / "eligibility_rules.yaml"))
    )

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
            print(f"[ERROR] Claude API degraded mode: {fallback_issues[-1]}")
        session_manager.save_state(session_id, updated_state, completed=False)

    def on_completed(updated_state: dict) -> None:
        report = updated_state.get("final_report", {})
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
    print("\n=== Final Decision ===")
    print(final_state.get("decision_status", "Requires More Info"))
    print(final_state.get("decision_summary", "No summary available"))


if __name__ == "__main__":
    main()
