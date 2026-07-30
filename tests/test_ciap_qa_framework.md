# CIAP QA Framework

## Overview
This suite focuses on functional correctness, boundary conditions, resilience, and conversation-state behavior for the CIAP application.

## Execution
Run from the repository root:

```bash
pytest -q tests/test_ciap_qa_framework.py
pytest --cov=. --cov-report=term-missing tests/test_ciap_qa_framework.py
```

## Coverage Focus
- Rule evaluation thresholds and boundary values
- Session start/resume behavior
- Extraction and validation robustness
- Follow-up and decision-agent state transitions
- Error handling and graceful degradation
