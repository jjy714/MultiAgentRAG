from typing import TypedDict, List


class PlanState(TypedDict):
    step: List[str]    # list of sub-question strings
    is_report: bool    # whether to produce a formal report at the end