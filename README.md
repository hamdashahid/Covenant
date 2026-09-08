# Covenant — CIAP Mortgage Eligibility Assistant

This README is the living handoff document for CIAP. Update it whenever the conversation design, eligibility policy, architecture, or current work changes so another contributor can continue without reconstructing the project history.

## Main purpose

CIAP is a terminal-based mortgage pre-check assistant. The assistant introduces itself as Alex and interviews an applicant one question at a time. It should understand ordinary human language, remember information already supplied, validate financial answers, and give a consistent preliminary eligibility result.

The intended experience is a short, mutual conversation with a professional mortgage officer—not a form read aloud and not a generic chatbot. CIAP must be warm and context-aware without making lending promises. Its result is an initial pre-check, not a mortgage approval.

## Current working flow

1. `main.py` starts or resumes a session and assembles the application.
2. `graph/ciap_graph.py` runs a LangGraph cycle: `interview → extract and validate → decide`.
3. Python chooses the next unanswered topic; the language model helps phrase it naturally.
4. One semantic-understanding step interprets the user's whole message and can collect several facts.
5. Deterministic validation accepts, rejects, clarifies, or defers the interpreted values.
6. Valid facts are merged into one conversation profile without losing earlier information.
7. The decision agent either requests the next missing answer or evaluates the profile.
8. Python applies the YAML eligibility policy; the language model does not invent the decision.
9. SQLite stores the session, messages, profile, and report for session resume.

The CLI currently calls `graph.invoke(...)`, so one process handles one conversation synchronously. It suits the terminal demo but is not yet a production service for many simultaneous users.

## Information currently collected

The active interview asks about:

1. Down payment
2. Credit score
3. Employment status
4. Time at the current job or business
5. Annual income
6. Total savings
7. Monthly debt payments

Before this sequence, Alex asks whether this is the applicant's first home. That answer is conversational context only and must not be treated as a down-payment answer.

Property value and requested loan amount were removed from the active flow because they are not required for the current demonstration. Some underlying schema and rule support remains for possible future use.

A retired applicant remains `retired`, skips the current-job-duration question, and is asked about current recurring pension, investment, rental, benefit, or other income. Salary earned before retirement must not be recorded as current income.

## Human-language behaviour

- `I have no down payment` means zero down payment.
- `I don't pay debt` means zero monthly debt.
- `freelancer`, `self emp`, and `business owner` mean self-employed.
- `emp` and clear full-time or part-time descriptions can mean employed.
- `I was laid off` means not currently employed; current recurring income can be explored.
- `retired officer` means retired, not automatically unemployed.
- `from 2021` is converted into elapsed years.
- Values such as `30k`, `90,000`, number words, ranges, and estimates are interpreted before validation.
- A message containing several facts should save every clearly stated fact and avoid duplicate questions.
- A correction such as `I meant 90,000, not 70,000` should replace the earlier value.

Conversation controls:

- `skip`, `move on`, and similar wording defer the current question and continue.
- `I don't want to tell you` is a refusal, not an invalid number.
- `I don't know` leaves a value unknown; Alex may offer one helpful range, then allow a skip.
- `What is debt?` receives a brief explanation before the question is asked again naturally.
- `stop` and `end` stop the interview.
- Gibberish such as `abcd` must not become zero, a skip, or an answer to another field.
- Negative money or duration and credit outside 300–850 receive a relevant clarification.
- Skipped and refused questions must not repeat forever; state tracks skipped fields, attempts, and reasons.

If OpenAI is unavailable because of credentials, exhausted credits, rate limits, or connectivity, CIAP uses a limited local reader. It handles common expressions but cannot understand language as broadly as the model, so its terminal warning should remain visible.

## Conversation style goal

Alex should sound like a concise professional chatting naturally:

- Use one or two short sentences and exactly one question per turn.
- Respond to meaning instead of repeating the user's answer.
- React only when it adds value; routine numbers do not always need praise.
- Vary the function of replies: acknowledge, explain relevance, ask a useful follow-up, or move directly ahead.
- Refer naturally to details already supplied, such as occupation or self-employment.
- Answer the user's question before continuing the interview.
- Show measured empathy for layoffs, uncertainty, or difficulty.
- Never expose field names, underscores, validation terms, logs, or internal messages.
- Never assume a currency the user did not provide.
- Never guarantee approval, an interest rate, loan amount, or timeline.

The main remaining design problem is conversational continuity. Earlier work improved greetings and sentence variation, but later turns can still reveal the underlying questionnaire. The next pass must connect each reply to the applicant's situation while preserving reliable state and deterministic validation.

## Work completed so far

The `minahil` branch has received several reliability and humanization passes:

- Built a one-question-at-a-time interview.
- Separated the first-home opening from down-payment extraction.
- Prevented `no` and unrelated opening text from silently becoming financial values.
- Added one semantic turn-understanding contract through the OpenAI adapter.
- Added deterministic validation after meaning is interpreted.
- Centralized skip, refusal, unknown, clarification, stop, and answer intent handling.
- Added multi-fact extraction and remembered volunteered information.
- Added profile correction and conflict handling.
- Improved employment, freelancer, layoff, and retirement interpretation.
- Added natural-zero handling for payment, savings, and debt phrases.
- Added numeric boundaries and year-to-duration interpretation.
- Removed property-value and loan-amount questions from the active policy.
- Added field attempts and deferral state to prevent question loops.
- Added contextual follow-ups and short explanations for concepts such as debt.
- Shortened and humanized greetings, transitions, questions, and final summaries.
- Kept eligibility decisions deterministic and separate from generated language.
- Added early termination for decisive profiles while handling retirement separately.
- Added SQLite session resume, profiles, messages, reports, runtime state, and tags.
- Ensured SQLite connections close and disk-full persistence errors do not crash the live interview.
- Added a visible degraded mode for OpenAI failures.
- Expanded regression coverage for failures found during repeated demos.

## Current eligibility policy

`rules/eligibility_rules.yaml` currently defines:

- Minimum annual income: 50,000
- Maximum debt-to-income ratio: 0.43
- Minimum credit score: 650
- Accepted statuses: employed, self-employed, and retired
- Minimum employment history: 2 years, except for the retired flow
- Minimum down-payment percentage support: 5% when property value exists

When property value is not collected, a recorded down payment currently passes that individual check only when it is positive. These are demonstration rules, not universal lending policy.

## Current status

Working now:

- The terminal interview runs from greeting to final result.
- OpenAI with `gpt-4o` is the configured provider/model.
- Sessions can be saved and resumed.
- Common natural answers, zero phrases, employment descriptions, skips, refusals, explanations, corrections, and invalid values are covered.
- Decisions return Eligible, Ineligible, or Requires More Info using deterministic rules.
- The tests provide broad unit, flow, QA, and conversational-regression coverage.

Still in progress:

- Make the middle and end feel like a connected conversation, not a reworded questionnaire.
- Test more mixed-language replies, typos, interruptions, emotional statements, and multi-intent answers.
- Decide whether total savings should stay in the interview; it is collected but is not required for eligibility.
- Review early termination so useful interviewing does not stop too soon.
- Build an async server and production storage before claiming concurrent-user support.
- Stop committing `.pytest-tmp-*` databases and add suitable ignore rules in a separate cleanup.
- Add a safe `.env.example`; none is currently present.
- Continue refining result wording without overstating what a lender will do.

## Main files and changes

| File | Responsibility and relevant work |
|---|---|
| `main.py` | Entry point, question policy, OpenAI setup, session lifecycle, graph assembly, early-stop settings, and persistence resilience. |
| `graph/ciap_graph.py` | LangGraph state and interview → extraction/validation → decision routing. |
| `agents/interview_agent.py` | Next-topic selection, opening behaviour, control intents, loop prevention, and contextual question wording. |
| `agents/extraction_validation.py` | Semantic facts to validated profile changes, corrections, invalid values, and follow-ups. |
| `agents/decision_agent.py` | Missing-data handling, early completion, final evaluation, and decision summary. |
| `llm/openai_adapter.py` | OpenAI semantic understanding and phrasing plus limited local fallback. |
| `core/conversation_intent.py` | Shared skip, refusal, unknown, clarification, stop, and answer classification. |
| `core/context_builder.py` | Extraction context from current state and conversation history. |
| `core/profile_updater.py` | Safe merging of new facts and corrections into the profile. |
| `core/response_planner.py` | Short, value-aware acknowledgements and transitions. |
| `core/schemas.py` | Supported fields, required eligibility fields, labels, and validation expectations. |
| `core/session_manager.py` | Session start, save, resume, runtime state, and completion. |
| `core/terminal_ui.py` | Terminal questions, warnings, results, and collected-information display. |
| `config/prompts_system_prompt.txt` | Alex's tone, one-question rule, context use, retirement flow, and safety boundaries. |
| `rules/eligibility_rules.yaml` | Editable policy thresholds. |
| `rules/rule_evaluator.py` | Deterministic evaluation and auditable rule trace. |
| `persistence/sqlite_store.py` | SQLite sessions, profiles, messages, reports, tags, and connection lifecycle. |
| `tests/test_conversational_regressions.py` | Regression tests for failures observed in demos. |
| `tests/test_human_language_matrix.py` | Varied human-language and financial-expression coverage. |
| `tests/test_extraction_validation.py` | Extraction, validation, corrections, and follow-up tests. |
| `tests/test_response_planner.py` | Contextual, short, non-repetitive response tests. |
| `tests/test_decision_agent.py` and `tests/test_rule_evaluator.py` | Required data, early stop, eligibility, and boundaries. |
| `tests/test_tags_and_flow.py` | Graph, persistence, resume, and tagging flows. |

`llm/claude_adapter.py` exists as an alternative but is not wired into `main.py`.

## Repositories and branches

- Working repository: `https://github.com/hamdashahid/Covenant.git`
- Working branch: `minahil`
- Second remote: `https://github.com/aiotac/ciap-v2`
- The second repository also has a `minahil` branch for reviewed work.

Check the active branch and remote state before transferring changes. Never commit `.env`, API keys, local databases, virtual environments, or generated test artifacts.

## Windows setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Create `.env` in the project root and do not commit it:

```dotenv
OPENAI_API_KEY=your_key_here
```

The key needs API credits; a ChatGPT subscription alone does not provide them. Check loading without printing the secret:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```

## Run and test

```powershell
python main.py
python main.py --session-id SESSION_ID
python main.py --list-tags
python -m pytest -q
```

Unittest discovery is also supported:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Tests must mock OpenAI or skip safely without credentials. Test collection must not create a real API client and fail only because a key is absent.

## Guidance for the next contributor

Do not fix isolated phrases without testing the complete flow. A local wording patch can break another topic when intent classification, state, extraction, and response generation disagree.

For each new issue:

1. Record the full conversation that exposes it.
2. Identify whether the fault is meaning, state, validation, response planning, or policy.
3. Fix the shared layer instead of adding a one-phrase exception when possible.
4. Add a regression test for the original wording and several paraphrases.
5. Run the full suite and manually test at least three different applicant personas.
6. Compare normal API mode with limited fallback mode.

Preserve this design principle:

> Understand the whole turn once, keep one reliable conversation state, validate deterministically, and use the language model only to express the next safe action naturally.

## Keeping this document current

After future work, update the completed work, current status, file map, policy, and setup sections. Do not paste API keys, applicant data, or private chat histories here; summarize decisions and reproducible examples instead.
