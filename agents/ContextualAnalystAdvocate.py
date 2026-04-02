import os
import asyncio
import random
import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
from graph.state import DiscussionState, AdvocateVote

load_dotenv()

AGENT_PROMPT = "agents/Prompts/ContextualAnalystAdvocate.yaml"


## LangGraph node function that advocates for routing the query to the medium tier
async def ContextualAnalystAdvocate(state: DiscussionState) -> dict:
    """
    args   : {
        "state (DiscussionState)": "discussion state containing 'question'"
    }
    return : {
        "dict": "updated state with 'votes' list containing one AdvocateVote and 'token_usage' dict"
    }
    """
    # Stagger the parallel requests to avoid 429 burst errors
    await asyncio.sleep(random.uniform(0.5, 2.0))

    question = state.get("question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt.format(question=question)),
        ]
    )

    # Parse the content into AdvocateVote
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    try:
        vote_data = json.loads(content)
        result = AdvocateVote(**vote_data)
    except Exception:
        result = AdvocateVote(role="contextual_analyst", vote="medium", reasoning="Fallback due to parsing error.")

    token_usage = normalize_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else {})

    print(
        f"[ContextualAnalystAdvocate] vote={result.get('vote')} | reason={result.get('reasoning')}"
    )
    return {
        "votes": [result],
        "token_usage": token_usage
    }

