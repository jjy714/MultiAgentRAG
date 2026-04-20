import asyncio
import random
import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import get_token_usage, strip_json_fence
from graph.state import DiscussionState, AdvocateVote

load_dotenv()

AGENT_PROMPT = "agents/Prompts/DirectResponderAdvocate.yaml"


async def DirectResponderAdvocate(state: DiscussionState) -> dict:
    # Stagger parallel requests to avoid 429 burst errors
    await asyncio.sleep(random.uniform(0.1, 1.0))

    question = state.get("question", "")
    response = await get_llm().ainvoke([
        SystemMessage(get_system_prompt(AGENT_PROMPT)),
        HumanMessage(get_user_prompt(AGENT_PROMPT).format(question=question)),
    ])

    try:
        vote = AdvocateVote(**json.loads(strip_json_fence(response.content)))
    except Exception:
        vote = AdvocateVote(role="direct_responder", vote="easy", reasoning="Fallback due to parsing error.")

    print(f"[DirectResponderAdvocate] vote={vote.get('vote')} | reason={vote.get('reasoning')}")
    return {"votes": [vote], "token_usage": get_token_usage(response)}
