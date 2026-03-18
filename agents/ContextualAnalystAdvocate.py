import os
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from graph.state import DiscussionState, AdvocateVote

load_dotenv()

AGENT_PROMPT = "agents/Prompts/ContextualAnalystAdvocate.yaml"


## LangGraph node function that advocates for routing the query to the medium RAG tier
def ContextualAnalystAdvocate(state: DiscussionState) -> dict:
    """
    args   : {
        "state (DiscussionState)": "discussion state containing 'question'"
    }
    return : {
        "dict": "updated state with 'votes' list containing one AdvocateVote"
    }
    """
    question = state.get("question", "")
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    structured_llm = get_llm().with_structured_output(AdvocateVote)

    result: AdvocateVote = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(question=question))
    ])
    print(f"[ContextualAnalystAdvocate] vote={result.get('vote')} | reason={result.get('reasoning')}")
    return {"votes": [result]}
