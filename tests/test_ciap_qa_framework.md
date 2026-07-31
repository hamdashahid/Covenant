# CIAP QA Framework

## Test Strategy
The CIAP QA suite is designed to validate the mortgage interview flow, business-rule evaluation, extraction validation, session persistence, graph orchestration, and resilience to malformed or hostile input. The strategy balances unit tests, component tests, and end-to-end conversation-style tests.

## Test Categories
- Functional conversation flow
- Boundary value analysis for business thresholds
- Input validation and robustness
- Error handling and recovery
- Decision-agent fallback behavior
- Persistence and state management
- Security-focused input handling
- Graph routing behavior
- CLI and startup orchestration

## Coverage Matrix
| Module | Coverage Focus | Status |
| --- | --- | --- |
| main.py | startup orchestration and graph integration | Covered |
| InterviewAgent | prompt generation and response handling | Covered |
| DecisionAgent | follow-up routing and final decision states | Covered |
| ExtractionValidationNode | extraction parsing, validation, conflicts | Covered |
| RuleEvaluator | threshold evaluation and edge-case robustness | Covered |
| ContextBuilder | prompt construction | Covered |
| ProfileUpdater | profile merge and conflict tracking | Covered |
| SessionManager | session resume and persistence coordination | Covered |
| SQLiteStore | create/read/update/delete persistence and recovery | Covered |
| Conversation Graph | routing between interview/extract/decision nodes | Covered |

## Requirement Traceability Matrix
| Requirement | Covered By |
| --- | --- |
| Interview flow completes and persists state | tests/test_ciap_qa_framework.py |
| Boundary thresholds for income, DTI, credit score, employment years, LTV, down payment | tests/test_ciap_qa_framework.py, tests/test_qa_extensions.py |
| Invalid input handling | tests/test_ciap_qa_framework.py, tests/test_qa_extensions.py |
| Security and prompt-injection robustness | tests/test_qa_extensions.py |
| Database persistence and recovery | tests/test_qa_extensions.py |
| Graceful exception handling | tests/test_ciap_qa_framework.py, tests/test_qa_extensions.py |

## Boundary Analysis
- Income threshold: 49,999 / 50,000 / 50,001
- Credit score: 649 / 650 / 651
- Debt-to-income ratio: 0.429 / 0.430 / 0.431
- Employment years: 1 / 2 / 3
- Loan-to-value ratio: 0.949 / 0.950 / 0.951
- Down payment percentage: 4.99% / 5.00% / 5.01%

## Risk Analysis
- High risk: malformed user data causing downstream evaluation exceptions.
- Medium risk: CLI and terminal interactions breaking in non-interactive or automated environments.
- Medium risk: hidden regressions in prompt injection or malformed payload handling.

## Assumptions
- The YAML eligibility rules remain the source of truth for thresholds.
- The system should continue the interview flow even when LLM responses are unavailable or malformed.

## Limitations
- Full LLM-backed conversation tests are intentionally simulated with stubs.
- Performance benchmarking is functional rather than long-duration load testing.

## Execution Instructions
- Run: pytest tests -q
- Run with coverage: pytest tests -q --cov=. --cov-report=term-missing
