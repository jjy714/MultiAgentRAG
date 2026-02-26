from langgraph.graph import START, END, StateGraph
from graph.state import RagState
from agents.retriever_node import retriever_node
from agents.ExtractorAgent import ExtractorAgent
from agents.QuestionAnsweringAgent import QuestionAnsweringAgent
from agents.WebSearchAgent import WebSearchAgent


def _finalize_medium(state: RagState) -> dict:
    """Map RagState final_raw_answer back to GraphState.final_answer."""
    raw = state.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    return {"final_answer": answer}


def create_medium_graph():
    """
    Medium tier (single-pass RAG):
      retriever_node → ExtractorAgent ─┐
                                        ├─→ QuestionAnsweringAgent → END
      WebSearchAgent ──────────────────┘
    
    Both retrieval and web search run, their outputs (documents + notes) 
    are merged as context for QuestionAnsweringAgent.
    """
    graph = StateGraph(RagState)

    graph.add_node("retrieve",    retriever_node)
    graph.add_node("extract",     ExtractorAgent)
    graph.add_node("web_search",  WebSearchAgent)
    graph.add_node("answer",      QuestionAnsweringAgent)

    # Both retrieval and web_search kick off from START in parallel
    graph.add_edge(START,        "retrieve")
    graph.add_edge(START,        "web_search")

    # retrieve → extract → wait to merge at answer
    graph.add_edge("retrieve",   "extract")
    graph.add_edge("extract",    "answer")

    # web_search feeds documents too; merges at answer via shared state
    graph.add_edge("web_search", "answer")

    graph.add_edge("answer",     END)

    return graph.compile()
