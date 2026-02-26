from typing import TypedDict, List, Annotated, Optional
import operator


class QAAnswerState(TypedDict):
    answer: str
    confidence: str   # "high" | "medium" | "low"
    sources: List[str]


class StepOutput(TypedDict):
    task: str
    answer: str
    notes: List[str]


class GraphState(TypedDict):
    # Shared across all tiers
    original_question: str
    complexity: str                                      # "easy" | "medium" | "complex"
    final_answer: Optional[str]

    # Medium & Complex tier
    documents: List[str]
    doc_ids: List[str]
    notes: List[str]

    # Complex tier
    plan: List[str]
    step_question: List[dict]
    step_output: Annotated[List[StepOutput], operator.add]
    step_notes: Annotated[List[str], operator.add]
    is_report: bool
    plan_summary: Optional[str]

    # Easy tier
    web_needed: bool
    web_result: Optional[str]
