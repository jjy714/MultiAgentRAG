from typing import TypedDict, List, Annotated, Optional
import operator
from graph.state.GraphState import sum_dicts


class PlanExecState(TypedDict):
    original_question: str
    plan: List[str]
    step_question: Annotated[List[dict], operator.add]
    step_output: Annotated[List[dict], operator.add]
    step_notes: Annotated[List[str], operator.add]
    stop: bool
    plan_summary: Optional[str]
    token_usage: Annotated[dict, sum_dicts]
