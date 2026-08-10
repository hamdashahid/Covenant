# CIAP Feature Report

This report summarizes the four feature enhancements implemented in this session:

- Tags
- Configurable Greetings
- User-initiated Stop Conversation
- Early Termination (mid-conversation)

---

## 1) Tags

### What it does
Adds lightweight tags to conversation messages so the app can annotate turns for tracing, filtering, or future analytics.

### Files changed / added
- [persistence/sqlite_store.py](persistence/sqlite_store.py)
- [tests/test_tags_and_flow.py](tests/test_tags_and_flow.py)

### Config / environment variables
No new environment variables were required for tags.

### Database schema changes
- Message storage now supports a tags field for each persisted message.
- Existing message rows are read and written with tag data included.

### How to test / use it manually
Example flow:
- Start the app normally.
- Send a few turns.
- Check persisted session data or inspect stored messages to confirm tags are present.

Example command:
```bash
python main.py
```

---

## 2) Configurable Greetings

### What it does
Lets the app show a configurable greeting on the first turn instead of using only a hard-coded intro.

### Files changed / added
- [agents/interview_agent.py](agents/interview_agent.py)
- [main.py](main.py)
- [config/greeting.txt](config/greeting.txt) (if present)

### Config / environment variables
- `CIAP_GREETING`
  - Default: empty / not set
  - If set, it overrides the greeting file content.

### Database schema changes
No database schema changes.

### How to test / use it manually
Set an environment variable before running:
```bash
set CIAP_GREETING="Welcome! I will help you check mortgage eligibility."
python main.py
```

Or place a greeting in the configured greeting file if used by the app.

---

## 3) User-initiated Stop Conversation

### What it does
Allows the user to stop the interview explicitly using commands such as `stop` or `end`.

### Files changed / added
- [agents/interview_agent.py](agents/interview_agent.py)
- [graph/ciap_graph.py](graph/ciap_graph.py)
- [core/terminal_ui.py](core/terminal_ui.py)
- [tests/test_tags_and_flow.py](tests/test_tags_and_flow.py)

### Config / environment variables
No new environment variables were required.

### Database schema changes
No database schema changes.

### How to test / use it manually
Run the app and type one of these during the conversation:
```text
stop
end
/stop
/end
```

Expected behavior:
- The session ends gracefully.
- The app marks the conversation as stopped by the user.

---

## 4) Early Termination (mid-conversation)

### What it does
If enough rules already pass, the app can offer to terminate the interview early and finalize the assessment instead of continuing with more questions.

### Files changed / added
- [agents/decision_agent.py](agents/decision_agent.py)
- [agents/interview_agent.py](agents/interview_agent.py)
- [graph/ciap_graph.py](graph/ciap_graph.py)
- [tests/test_tags_and_flow.py](tests/test_tags_and_flow.py)

### Config / environment variables
- `EARLY_OFFER_PASS_RATIO`
  - Default: `0.85`
  - If the pass ratio meets or exceeds this value, the app can offer early termination.

- `EARLY_AUTO_PASS_RATIO`
  - Default: `1.0`
  - If the pass ratio meets or exceeds this value, the app can auto-complete without asking for confirmation.

### Database schema changes
No database schema changes.

### How to test / use it manually
Example:
- Provide enough information that the current profile already passes a high percentage of eligibility rules.
- The app should offer to finalize now.
- Respond with `yes` to confirm, or `no` to continue.

Example command:
```bash
set EARLY_OFFER_PASS_RATIO=0.85
set EARLY_AUTO_PASS_RATIO=1.0
python main.py
```

### Note
These thresholds are relative to the total number of rules in the evaluator and should be interpreted accordingly.

---

## Finalize-phrase handling (important behavior)

The app also now supports natural finalize phrases such as:
- `No, that's all the information I have`
- `That's all`
- `Thanks`

When the user says one of these, the app routes to decision evaluation instead of silently ending the conversation or continuing with generic follow-up questions.

---

## Test coverage summary

### Tests added / updated
The implementation added and updated regression coverage around:
- tags persistence and message storage
- stop flow handling
- finalize-phrase routing
- early termination confirmation
- combined multi-field input followed by finalize

### Final test suite result
- Total tests collected: 119
- Final result: 119 passed

Example verification command:
```bash
python -m pytest -q
```

---

## Known limitations / notes

- Early termination thresholds are only meaningful relative to the total number of evaluated rules.
- Natural-language finalize phrases are supported in a focused way and may need expansion if more conversational variants are desired.
- The app still relies on the current rule evaluator and extraction quality; if the LLM misses fields, the decision path will report missing information rather than silently finishing.
