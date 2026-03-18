import os
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from graph.state import GraphState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/DirectResponderAgent.yaml"


### Typed output schema for the DirectResponderAgent's structured LLM response
class DirectResponderOutput(TypedDict):
    """
    args   : {
        "answer (str)": "the agent's direct answer to the question",
        "web_needed (bool)": "whether a web search is required for more detail"
    }
    return : {
        "DirectResponderOutput": "structured dict with 'answer' and 'web_needed' fields"
    }
    """
    answer: str
    web_needed: bool


## LangGraph node function that answers easy questions directly from LLM knowledge
def DirectResponderAgent(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "graph state containing 'original_question'"
    }
    return : {
        "dict": "updated state with 'final_answer' (str) and 'web_needed' (bool)"
    }
    """
    question = state.get("original_question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    structured_llm = get_llm().with_structured_output(DirectResponderOutput)

    result: DirectResponderOutput = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=question))
    ])
    print(f"[DirectResponderAgent] web_needed={result.get('web_needed')}")
    return {
        "final_answer": result.get("answer", ""),
        "web_needed": result.get("web_needed", False),
    }
