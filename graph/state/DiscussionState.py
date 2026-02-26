from typing import TypedDict, Annotated, List
import operator


class AdvocateVote(TypedDict):
    role: str       # "direct_responder" | "contextual_analyst" | "deep_researcher"
    vote: str       # "easy" | "medium" | "complex"
    reasoning: str  # 1-2 sentence justification


class DiscussionState(TypedDict):
    question: str
    votes: Annotated[List[AdvocateVote], operator.add]  # parallel fan-in via reducer
    complexity: str  # final decision set by ModeratorAgent
