import os
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
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
        "dict": "updated state with 'plan' (List[str]) and 'is_report' (bool)"
    }
    """
    original_question = state.get("original_question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    structured_llm = get_llm().with_structured_output(PlanState)

    result: PlanState = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=original_question))
    ])
    print(f"[PlannerAgent] plan steps={len(result.get('step', []))} | is_report={result.get('is_report')}")
    return {
        "plan": result.get("step", []),
        "is_report": result.get("is_report", False),
    }
