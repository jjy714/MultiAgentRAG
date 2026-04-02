from typing import TypedDict, Annotated, List
import operator
from graph.state.GraphState import sum_dicts


### Typed dict representing a single advocate agent's vote in the discussion panel
class AdvocateVote(TypedDict):
    """
    args   : {
        "role (str)": "advocate identity — 'direct_responder', 'contextual_analyst', or 'deep_researcher'",
        "vote (str)": "proposed routing tier — 'easy', 'medium', or 'complex'",
        "reasoning (str)": "1–2 sentence justification for the vote"
    }
    return : {
        "AdvocateVote": "typed dict containing the advocate's voting result"
    }
    """
    role: str       # "direct_responder" | "contextual_analyst" | "deep_researcher"
    vote: str       # "easy" | "medium" | "complex"
    reasoning: str  # 1-2 sentence justification


### Shared state for the discussion panel graph that collects advocate votes
class DiscussionState(TypedDict):
    """
    args   : {
        "question (str)": "the user's original query",
        "votes (List[AdvocateVote])": "accumulated votes from advocate agents",
        "complexity (str)": "final routing decision set by the ModeratorAgent",
        "token_usage (dict)": "accumulated token usage metadata"
    }
    return : {
        "DiscussionState": "typed dict for the discussion panel subgraph"
    }
    """
    question: str
    votes: Annotated[List[AdvocateVote], operator.add]  # parallel fan-in via reducer
    complexity: str  # final decision set by ModeratorAgent
    token_usage: Annotated[dict, sum_dicts]
