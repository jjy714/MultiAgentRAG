from typing import TypedDict, List


class PlanState(TypedDict):
    step: List[str]
    is_report: bool
    web_needed: bool
