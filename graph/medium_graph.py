from langgraph.graph import START, END, StateGraph
from graph.state import RagState
from agents.retriever_node import retriever_node
from agents.ExtractorAgent import ExtractorAgent
from agents.QuestionAnsweringAgent import QuestionAnsweringAgent
from agents.WebSearchAgent import WebSearchAgent


def _finalize_medium(state: RagState) -> dict:
    """Promote the structured QA output to the top-level final_answer field."""
    raw = state.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    return {"final_answer": answer}


def create_medium_graph():
    graph = StateGraph(RagState)

    graph.add_node("retrieve", retriever_node)
    graph.add_node("extract", ExtractorAgent)
    graph.add_node("web_search", WebSearchAgent)
    graph.add_node("answer", QuestionAnsweringAgent)

    graph.add_edge(START, "retrieve")
    graph.add_edge(START, "web_search")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "answer")
    graph.add_edge("web_search", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
