from typing import TypedDict, List, Annotated, Optional
import operator

def sum_dicts(d1: dict, d2: dict) -> dict:
    """Combines two token usage dicts by summing their values."""
    if not d1: return d2
    if not d2: return d1
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}

### Structured output schema for a QA agent's answer
class QAAnswerState(TypedDict):
    """
    args   : {
        "answer (str)": "the answer text",
        "confidence (str)": "confidence level — 'high', 'medium', or 'low'",
        "sources (List[str])": "list of source references"
    }
    return : {
        "QAAnswerState": "typed dict with answer quality metadata"
    }
    """
    answer: str
    confidence: str   # "high" | "medium" | "low"
    sources: List[str]


### Typed dict representing a single completed step output in the plan executor
class StepOutput(TypedDict):
    """
    args   : {
        "task (str)": "the sub-question or task description",
        "answer (str)": "the generated answer for the step",
        "notes (List[str])": "extracted notes from retrieved documents"
    }
    return : {
        "StepOutput": "typed dict with per-step execution results"
    }
    """
    task: str
    answer: str
    notes: List[str]


### Main shared graph state passed across all tiers of the MultiAgentRAG pipeline
class GraphState(TypedDict):
    """
    args   : {
        "original_question (str)": "the user's original query",
        "complexity (str)": "routing decision — 'easy', 'medium', or 'complex'",
        "final_answer (Optional[str])": "the generated final answer",
        "documents (List[str])": "retrieved document contents",
        "doc_ids (List[str])": "corresponding document chunk IDs",
        "notes (List[str])": "extracted notes from documents",
        "plan (List[str])": "ordered list of sub-questions for complex tier",
        "step_question (List[dict])": "StepTaskState dicts for each planned step",
        "step_output (List[StepOutput])": "accumulated per-step results",
        "step_notes (List[str])": "accumulated extracted notes from all steps",
        "is_report (bool)": "whether to produce a formal report output",
        "plan_summary (Optional[str])": "synthesized summary across all steps",
        "web_needed (bool)": "whether web search is required (easy tier)",
        "web_result (Optional[str])": "raw web search result (easy tier)",
        "token_usage (dict)": "accumulated token usage metadata"
    }
    return : {
        "GraphState": "typed dict encompassing the full pipeline state"
    }
    """
    # Shared across all tiers
    original_question: str
    complexity: str                                      # "easy" | "medium" | "complex"
    final_answer: Optional[str]
    token_usage: Annotated[dict, sum_dicts]

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
