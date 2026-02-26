import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from graph.state import GraphState

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_NAME = os.getenv("VLLM_NAME")
AGENT_PROMPT = "agents/Prompts/DirectResponderAgent.yaml"


class DirectResponderOutput(TypedDict):
    answer: str
    web_needed: bool


def DirectResponderAgent(state: GraphState) -> dict:
    """Answers easy questions directly from LLM knowledge; flags if web search is needed."""
    question = state.get("original_question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = ChatOpenAI(
        model_name=VLLM_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    structured_llm = llm.with_structured_output(DirectResponderOutput)

    result: DirectResponderOutput = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=question))
    ])
    print(f"[DirectResponderAgent] web_needed={result.get('web_needed')}")
    return {
        "final_answer": result.get("answer", ""),
        "web_needed": result.get("web_needed", False),
    }
