# Covenant (CIAP)

Terminal-based conversational AI for mortgage eligibility assessment.

## Architecture

CIAP is orchestrated with LangGraph and uses three core nodes:

1. **Interview Agent** (`agents/interview_agent.py`)  
   Asks one mortgage question per turn in terminal mode (stdin/stdout), following a configurable policy.
2. **Extraction & Validation Node** (`agents/extraction_validation.py`)  
   Performs a single LLM extraction call per turn, validates/coerces fields against schema, and assigns confidence.
3. **Decision Agent** (`agents/decision_agent.py`)  
   Checks if enough validated data is available. If not, routes back for follow-up. If yes, evaluates deterministic eligibility rules via `rules/rule_evaluator.py`.

Supporting modules:
- Session manager (`core/session_manager.py`) for start/resume and model/session tracking.
- Context builder (`core/context_builder.py`) for LLM prompt assembly.
- Profile updater (`core/profile_updater.py`) for merged profile updates + conflict detection.
- Claude LLM adapter (`llm/claude_adapter.py`) with graceful fallback extraction.
- SQLite persistence (`persistence/sqlite_store.py`) for sessions, profiles, and reports in `core.db`.

## Rules

Rules are loaded from `rules/eligibility_rules.yaml` and include:
- Minimum annual income
- Maximum debt-to-income ratio
- Minimum credit score
- Allowed employment statuses

Eligibility decisions are always deterministic and auditable (non-LLM).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env` if you want live Claude extraction. Without a key, CIAP uses graceful fallback extraction.

## Run

```bash
python main.py
```

Resume a session:

```bash
python main.py --session-id <existing-session-id>
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Final Output

At the end of the interview loop, CIAP prints one of:
- `Eligible`
- `Ineligible`
- `Requires More Info`

Along with a short decision summary.
