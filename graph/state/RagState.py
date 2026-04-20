from typing import TypedDict, List, Annotated, Optional
import operator
from graph.state.GraphState import sum_dicts


class RagState(TypedDict):
    question: str
    documents: Annotated[List[str], operator.add]
    doc_ids: Annotated[List[str], operator.add]
    notes: Annotated[List[str], operator.add]
    final_raw_answer: Optional[dict]
    token_usage: Annotated[dict, sum_dicts]
