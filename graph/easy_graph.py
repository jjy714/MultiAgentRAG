from langgraph.graph import START, END, StateGraph
from graph.state import GraphState
from agents.DirectResponderAgent import DirectResponderAgent
from agents.WebSearchAgent import WebSearchAgent


## Conditional edge function that routes to web search if flagged, otherwise ends
def _route_web(state: GraphState) -> str:
    """
    args   : {
        "state (GraphState)": "graph state with 'web_needed' (bool)"
    }
    return : {
        "str": "'web_search' if web search is needed, otherwise END"
    }
    """
    return "web_search" if state.get("web_needed", False) else END


## LangGraph node that runs web search and merges its result with the initial answer
async def _web_search_then_finalize(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "graph state with 'original_question' and 'final_answer'"
    }
    return : {
        "dict": "updated state with merged 'final_answer' (str) and 'web_result' (str)"
    }
    """
    question = state.get("original_question", "")
    web_result = await WebSearchAgent({"question": question})
    web_docs = web_result.get("documents", [""])
    web_content = web_docs[0] if web_docs else ""
    # Merge initial answer with web search result
    merged_answer = f"{state.get('final_answer', '')}\n\n[Web Context]\n{web_content}".strip()
    return {"final_answer": merged_answer, "web_result": web_content}


## Build and compile the easy-tier graph with optional web search fallback
def create_easy_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled easy-tier graph ready for invocation"
    }
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
