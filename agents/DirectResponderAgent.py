import json
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import get_token_usage, strip_json_fence
from graph.state import GraphState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/DirectResponderAgent.yaml"


class DirectResponderOutput(TypedDict):
    answer: str
    web_needed: bool


def DirectResponderAgent(state: GraphState) -> dict:
    question = state.get("original_question", "")
    response = get_llm().invoke([
        SystemMessage(get_system_prompt(AGENT_PROMPT)),
        HumanMessage(get_user_prompt(AGENT_PROMPT).format(question=question))
    ])

    try:
        result = DirectResponderOutput(**json.loads(strip_json_fence(response.content)))
    except Exception:
        result = DirectResponderOutput(answer=response.content, web_needed=False)

    print(f"[DirectResponderAgent] web_needed={result.get('web_needed')}")
    return {
        "final_answer": result.get("answer", ""),
        "web_needed": result.get("web_needed", False),
        "token_usage": get_token_usage(response)
    }
