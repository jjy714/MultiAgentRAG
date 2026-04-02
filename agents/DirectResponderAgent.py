import os
import json
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
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
        "dict": "updated state with 'final_answer' (str), 'web_needed' (bool), and 'token_usage'"
    }
    """
    question = state.get("original_question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=question))
    ])

    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    try:
        data = json.loads(content)
        result = DirectResponderOutput(**data)
    except Exception:
        result = DirectResponderOutput(answer=response.content, web_needed=False)

    token_usage = normalize_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else {})

    print(f"[DirectResponderAgent] web_needed={result.get('web_needed')}")
    return {
        "final_answer": result.get("answer", ""),
        "web_needed": result.get("web_needed", False),
        "token_usage": token_usage
    }
