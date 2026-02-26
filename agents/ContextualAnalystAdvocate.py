import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from graph.state import DiscussionState, AdvocateVote

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_NAME = os.getenv("VLLM_NAME")
AGENT_PROMPT = "agents/Prompts/ContextualAnalystAdvocate.yaml"


def ContextualAnalystAdvocate(state: DiscussionState) -> dict:
    """Advocates that the query needs RAG retrieval (medium tier)."""
    question = state.get("question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = ChatOpenAI(
        model_name=VLLM_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    structured_llm = llm.with_structured_output(AdvocateVote)

    result: AdvocateVote = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=question))
    ])
    print(f"[ContextualAnalystAdvocate] vote={result.get('vote')} | reason={result.get('reasoning')}")
    return {"votes": [result]}
