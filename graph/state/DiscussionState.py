from typing import TypedDict, Annotated, List
import operator
from graph.state.GraphState import sum_dicts


class AdvocateVote(TypedDict):
    role: str      # "direct_responder" | "contextual_analyst" | "deep_researcher"
    vote: str      # "easy" | "medium" | "complex"
    reasoning: str


class DiscussionState(TypedDict):
    question: str
    votes: Annotated[List[AdvocateVote], operator.add]
    complexity: str
    token_usage: Annotated[dict, sum_dicts]
