from typing import TypedDict, List, Annotated, Optional
import operator


class PlanExecState(TypedDict):
    original_question: str
    plan: List[str]                                         # ordered list of sub-question strings
    step_question: List[dict]                               # StepTaskState dicts for each step
    step_output: Annotated[List[dict], operator.add]        # accumulated step results
    step_notes: Annotated[List[str], operator.add]          # accumulated extracted notes
    stop: bool                                              # true when all steps done
    plan_summary: Optional[str]
