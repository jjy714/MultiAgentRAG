from langgraph.graph import START, END, StateGraph
from graph.state import RagState
from agents.retriever_node import retriever_node
from agents.ExtractorAgent import ExtractorAgent
from agents.QuestionAnsweringAgent import QuestionAnsweringAgent
from agents.WebSearchAgent import WebSearchAgent


## Map the RagState final_raw_answer back to the GraphState final_answer field
def _finalize_medium(state: RagState) -> dict:
    """
    args   : {
        "state (RagState)": "RAG state containing 'final_raw_answer' (dict)"
    }
    return : {
        "dict": "updated state with 'final_answer' (str) extracted from QAAnswerState"
    }
    """
    raw = state.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    return {"final_answer": answer}


## Build and compile the medium-tier single-pass RAG graph with parallel retrieval and web search
def create_medium_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled medium-tier graph ready for invocation"
    }
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
