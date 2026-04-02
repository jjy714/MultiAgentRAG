from typing import TypedDict, List, Annotated, Optional
import operator
from graph.state.GraphState import sum_dicts


### State for the plan-executor loop that manages step-by-step execution of a complex query
class PlanExecState(TypedDict):
    """
    args   : {
        "original_question (str)": "the user's original query",
        "plan (List[str])": "ordered list of sub-question strings",
        "step_question (List[dict])": "StepTaskState dicts for each planned step",
        "step_output (List[dict])": "accumulated per-step results",
        "step_notes (List[str])": "accumulated extracted notes across all steps",
        "stop (bool)": "flag set to True when all steps are complete",
        "plan_summary (Optional[str])": "synthesized final answer across all steps",
        "token_usage (dict)": "accumulated token usage metadata"
    }
    return : {
        "PlanExecState": "typed dict for the plan executor subgraph"
    }
    """
    original_question: str
    plan: List[str]                                         # ordered list of sub-question strings
    step_question: List[dict]                               # StepTaskState dicts for each step
    step_output: Annotated[List[dict], operator.add]        # accumulated step results
    step_notes: Annotated[List[str], operator.add]          # accumulated extracted notes
    stop: bool                                              # true when all steps done
    plan_summary: Optional[str]
    token_usage: Annotated[dict, sum_dicts]
