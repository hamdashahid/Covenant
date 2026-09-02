from core.response_planner import build_response_plan


def test_routine_turns_rotate_between_direct_and_brief_modes() -> None:
    modes = {
        build_response_plan({"turn_count": turn}, "credit_score").mode
        for turn in range(3)
    }
    assert modes == {"direct_transition", "brief_transition", "contextual_transition"}


def test_layoff_receives_empathetic_transition_without_echo_permission() -> None:
    plan = build_response_plan(
        {"latest_user_response": "I was laid off last month"},
        "employment_years",
    )
    assert plan.mode == "empathetic_transition"
    assert plan.allow_value_echo is False


def test_correction_is_confirmed_but_not_praised() -> None:
    plan = build_response_plan(
        {"recent_profile_corrections": ["annual_income"]},
        "monthly_debt",
    )
    assert plan.mode == "correction"
    assert "confirmation" in plan.instruction.lower()
    assert "praise" in plan.instruction.lower()
    assert plan.allow_value_echo is False


def test_clarification_has_priority_over_routine_acknowledgement() -> None:
    plan = build_response_plan(
        {"clarification_context": "what does debt mean?"},
        "monthly_debt",
        is_followup=True,
    )
    assert plan.mode == "clarification"
    assert plan.allow_value_echo is True
