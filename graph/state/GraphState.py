from typing import TypedDict, List, Annotated, Optional
import operator


def sum_dicts(d1: dict, d2: dict) -> dict:

    if not d1:
        return d2
    if not d2:
        return d1
    result = {}
    for k in set(d1) | set(d2):
        v1 = d1.get(k, 0)
        v2 = d2.get(k, 0)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            result[k] = v1 + v2
        elif isinstance(v1, (int, float)):
            result[k] = v1
        elif isinstance(v2, (int, float)):
            result[k] = v2
    return result


class QAAnswerState(TypedDict):
    answer: str
    confidence: str  # "high" | "medium" | "low"
    sources: List[str]


class StepOutput(TypedDict):
    task: str
    answer: str
    notes: List[str]


class GraphState(TypedDict):
    original_question: str
    complexity: str  # "easy" | "medium" | "complex"
    final_answer: Optional[str]
    token_usage: Annotated[dict, sum_dicts]

    documents: List[str]
    doc_ids: List[str]
    notes: List[str]

    plan: List[str]
    step_question: Annotated[List[dict], operator.add]
    step_output: Annotated[List[StepOutput], operator.add]
    step_notes: Annotated[List[str], operator.add]
    is_report: bool
    plan_summary: Optional[str]

    web_needed: bool
    web_result: Optional[str]
