import os
import asyncio
import random
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
from graph.state import DiscussionState, AdvocateVote

load_dotenv()

AGENT_PROMPT = "agents/Prompts/DirectResponderAdvocate.yaml"


## LangGraph node function that advocates for routing the query to the easy tier
async def DirectResponderAdvocate(state: DiscussionState) -> dict:
    """
    args   : {
        "state (DiscussionState)": "discussion state containing 'question'"
    }
    return : {
        "dict": "updated state with 'votes' list containing one AdvocateVote and 'token_usage' dict"
    }
    """
    # Stagger the parallel requests to avoid 429 burst errors
    await asyncio.sleep(random.uniform(0.1, 1.0))
    
    question = state.get("question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    # Use standard LLM to get token usage metadata, then parse to structured output
    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt.format(question=question)),
        ]
    )
    
    # Parse the content into AdvocateVote (since it's JSON)
    import json
    # Extract the JSON block if the LLM wrapped it in markdown
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    try:
        vote_data = json.loads(content)
        result = AdvocateVote(**vote_data)
    except Exception:
        # Fallback if parsing fails
        result = AdvocateVote(role="direct_responder", vote="easy", reasoning="Fallback due to parsing error.")

    token_usage = normalize_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else {})

    print(
        f"[DirectResponderAdvocate] vote={result.get('vote')} | reason={result.get('reasoning')}"
    )
    return {
        "votes": [result],
        "token_usage": token_usage
    }
