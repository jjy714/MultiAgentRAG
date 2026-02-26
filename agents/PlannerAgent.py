import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from graph.state import GraphState, PlanState

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_NAME = os.getenv("VLLM_NAME")
AGENT_PROMPT = "agents/Prompts/PlannerAgent.yaml"


def PlannerAgent(state: GraphState) -> dict:
    """Breaks down a complex query into an ordered list of sub-questions."""
    original_question = state.get("original_question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = ChatOpenAI(
        model_name=VLLM_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    structured_llm = llm.with_structured_output(PlanState)

    result: PlanState = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=original_question))
    ])
    print(f"[PlannerAgent] plan steps={len(result.get('step', []))} | is_report={result.get('is_report')}")
    return {
        "plan": result.get("step", []),
        "is_report": result.get("is_report", False),
    }
