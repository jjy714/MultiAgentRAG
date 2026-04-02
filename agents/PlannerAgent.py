import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
from graph.state import GraphState, PlanState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/PlannerAgent.yaml"


## LangGraph node function that decomposes a complex query into an ordered list of sub-questions
def PlannerAgent(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "graph state containing 'original_question' (str)"
    }
    return : {
        "dict": "updated state with 'plan' (List[str]), 'is_report' (bool), and 'token_usage'"
    }
    """
    original_question = state.get("original_question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=original_question))
    ])
    
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
        result = PlanState(**data)
    except Exception:
        result = PlanState(step=[], is_report=False)

    token_usage = normalize_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else {})

    print(f"[PlannerAgent] plan steps={len(result.get('step', []))} | is_report={result.get('is_report')}")
    return {
        "plan": result.get("step", []),
        "is_report": result.get("is_report", False),
        "token_usage": token_usage
    }
