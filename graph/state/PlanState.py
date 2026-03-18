from typing import TypedDict, List


### Structured output schema for the PlannerAgent's query decomposition result
class PlanState(TypedDict):
    """
    args   : {
        "step (List[str])": "ordered list of sub-question strings",
        "is_report (bool)": "whether to produce a formal report at the end"
    }
    return : {
        "PlanState": "typed dict with the decomposed plan and report flag"
    }
    """
    step: List[str]    # list of sub-question strings
    is_report: bool    # whether to produce a formal report at the end