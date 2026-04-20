import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import get_token_usage, strip_json_fence
from graph.state import GraphState, PlanState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/PlannerAgent.yaml"


def PlannerAgent(state: GraphState) -> dict:
    original_question = state.get("original_question", "")
    response = get_llm().invoke([
        SystemMessage(get_system_prompt(AGENT_PROMPT)),
        HumanMessage(get_user_prompt(AGENT_PROMPT).format(question=original_question))
    ])

    try:
        plan = PlanState(**json.loads(strip_json_fence(response.content)))
    except Exception:
        plan = PlanState(step=[], is_report=False)

    steps = plan.get("step", [])[:5]
    web_needed = plan.get("web_needed", False)
    print(f"[PlannerAgent] plan steps={len(steps)} | is_report={plan.get('is_report')} | web_needed={web_needed}")
    return {
        "plan": steps,
        "is_report": plan.get("is_report", False),
        "web_needed": web_needed,
        "token_usage": get_token_usage(response)
    }
