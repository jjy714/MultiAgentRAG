from typing import TypedDict, List, Annotated, Optional
from graph.state.GraphState import sum_dicts


### Shared state for single-pass RAG subgraphs (medium tier and individual complex steps)
class RagState(TypedDict):
    """
    args   : {
        "question (str)": "the sub-question or main query to answer",
        "documents (List[str])": "retrieved document contents",
        "doc_ids (List[str])": "corresponding document chunk IDs",
        "notes (List[str])": "extracted notes from documents",
        "final_raw_answer (Optional[dict])": "structured QAAnswerState output",
        "token_usage (dict)": "accumulated token usage metadata"
    }
    return : {
        "RagState": "typed dict for RAG pipeline subgraph state"
    }
    """
    question: str
    documents: List[str]
    doc_ids: List[str]
    notes: List[str]
    final_raw_answer: Optional[dict]   # QAAnswerState
    token_usage: Annotated[dict, sum_dicts]
