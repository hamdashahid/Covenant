from agents.interview_agent import InterviewAgent
from agents.decision_agent import DecisionAgent
import core.terminal_ui as terminal_ui

class DummyEval:
    def __init__(self):
        self.rules = {"income_threshold": 100000, "min_credit_score": 640}
    def evaluate(self, profile):
        return {"status": "Ineligible", "summary": "test", "rule_breakdown": []}

terminal_ui.print_agent_message = lambda *a, **k: None
terminal_ui.print_thinking = lambda *a, **k: None
terminal_ui.get_answer_prompt = lambda: "Hi"
agent = InterviewAgent([], "sys", llm_client=None)
state = agent({"conversation_history": []})
print("greeting", state.get("user_requested_stop"), state.get("skip_extraction"), state.get("needs_followup"))

terminal_ui.get_answer_prompt = lambda: "I do not want to continue this conversation."
agent2 = InterviewAgent([], "sys", llm_client=None)
state2 = agent2({"conversation_history": []})
print("stop", state2.get("user_requested_stop"), state2.get("decision_status"))

agent3 = DecisionAgent(DummyEval())
state3 = agent3({"applicant_profile": {"annual_income": 50000, "credit_score": 600}, "max_turns": 8, "turn_count": 0})
print("early", state3.get("decision_status"), state3.get("needs_followup"), state3.get("final_report", {}).get("status"))
