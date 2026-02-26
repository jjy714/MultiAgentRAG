from langgraph.graph import START, END, StateGraph
from graph.state import GraphState
from agents.DirectResponderAgent import DirectResponderAgent
from agents.WebSearchAgent import WebSearchAgent


def _route_web(state: GraphState) -> str:
    """After DirectResponder: go to web search if needed, else finish."""
    return "web_search" if state.get("web_needed", False) else END


async def _web_search_then_finalize(state: GraphState) -> dict:
    """Run WebSearchAgent and append its result to the final answer."""
    question = state.get("original_question", "")
    web_result = await WebSearchAgent({"question": question})
    web_docs = web_result.get("documents", [""])
    web_content = web_docs[0] if web_docs else ""
    # Merge initial answer with web search result
    merged_answer = f"{state.get('final_answer', '')}\n\n[Web Context]\n{web_content}".strip()
    return {"final_answer": merged_answer, "web_result": web_content}


def create_easy_graph():
    """
    Easy tier:
      DirectResponderAgent → [if web_needed] WebSearchAgent → END
                           → [else]                         → END
    """
    graph = StateGraph(GraphState)

    graph.add_node("direct_responder", DirectResponderAgent)
    graph.add_node("web_search",       _web_search_then_finalize)

    graph.add_edge(START, "direct_responder")
    graph.add_conditional_edges(
        "direct_responder",
        _route_web,
        {"web_search": "web_search", END: END},
    )
    graph.add_edge("web_search", END)

    return graph.compile()
