# Covenant (CIAP)

Terminal-based conversational AI for mortgage eligibility assessment.

## Architecture

End-to-end flow:

`Terminal Input + Configuration → Session Manager → Context Builder → LangGraph [Interview Agent → Extraction & Validation Node → Decision Agent (routes back to Interview Agent when data is incomplete)] → LLM Client Adapter → Profile Updater → Rule Evaluator (deterministic) → Persistence Layer (write + close session) → Final Decision printed to stdout`

Key implementation details:
- **Single LLM provider/model:** Anthropic Claude Messages API only, using `claude-sonnet-4-6`.
- **Interview system prompt config:** `config/prompts_system_prompt.txt`.
- **Single validation point:** extraction + schema coercion happens in `agents/extraction_validation.py`; Decision Agent consumes validated profile and does not re-validate fields.
- **Deterministic eligibility decision:** `rules/rule_evaluator.py` applies `rules/eligibility_rules.yaml` (non-LLM).
- **Graceful degradation:** Claude API failures retry with backoff, then fallback extraction is used with a clear terminal error state.
- **SQLite persistence (`core.db`) tables:**
  - `sessions` (`session_id`, `model_id`, `session_state`, `created_at`, `closed_at`)
  - `applicant_profiles`
  - `eligibility_reports`
  - `messages` (`message_id`, `session_id`, `role`, `content`, `message_timestamp`)

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

Set `ANTHROPIC_API_KEY` in `.env` for Claude API access. If API calls fail, CIAP retries with backoff, then enters a visible degraded extraction mode.

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
