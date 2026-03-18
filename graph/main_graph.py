from langgraph.graph import START, END, StateGraph
from graph.state import GraphState, DiscussionState
from graph.discussion_graph import create_discussion_graph
from graph.easy_graph import create_easy_graph
from graph.medium_graph import create_medium_graph
from graph.complex_graph import create_complex_graph


# ─── Discussion Panel node ──────────────────────────────────────────────────────

## LangGraph node that runs the 3-advocate discussion panel and writes complexity to GraphState
async def discussion_panel_node(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "graph state containing 'original_question' (str)"
    }
    return : {
        "dict": "updated state with 'complexity' (str) set to 'easy', 'medium', or 'complex'"
    }
    """
    discussion_graph = create_discussion_graph()
    discussion_input: DiscussionState = {
        "question": state.get("original_question", ""),
        "votes": [],
        "complexity": "",
    }
    result = await discussion_graph.ainvoke(discussion_input)
    complexity = result.get("complexity", "medium")
    print(f"\n{'='*50}")
    print(f"[DiscussionPanel] Routing to tier: '{complexity}'")
    print(f"{'='*50}\n")
    return {"complexity": complexity}


# ─── Tier wrapper nodes ─────────────────────────────────────────────────────────

## LangGraph node that delegates the query to the easy-tier subgraph
async def easy_node(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "full graph state"
    }
    return : {
        "dict": "updated state with 'final_answer' (str) from the easy tier"
    }
    """
    easy_graph = create_easy_graph()
    return await easy_graph.ainvoke(state)


## LangGraph node that delegates the query to the medium-tier single-pass RAG subgraph
async def medium_node(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "full graph state containing 'original_question'"
    }
    return : {
        "dict": "updated state with 'final_answer' (str) from the medium tier"
    }
    """
    medium_graph = create_medium_graph()
    medium_input = {
        "question": state.get("original_question", ""),
        "documents": [],
        "doc_ids": [],
        "notes": [],
        "final_raw_answer": None,
    }
    result = await medium_graph.ainvoke(medium_input)
    raw = result.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    return {"final_answer": answer}


## LangGraph node that delegates the query to the complex multi-step plan-executor subgraph
async def complex_node(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "full graph state containing 'original_question' and 'plan'"
    }
    return : {
        "dict": "updated state with 'final_answer' (str) from the complex tier"
    }
    """
    complex_graph = create_complex_graph()
    return await complex_graph.ainvoke(state)


# ─── Router ─────────────────────────────────────────────────────────────────────

## Conditional edge function that routes the graph to the correct tier based on complexity
def route_by_complexity(state: GraphState) -> str:
    """
    args   : {
        "state (GraphState)": "graph state with 'complexity' (str)"
    }
    return : {
        "str": "tier name — 'easy', 'medium', or 'complex'"
    }
    """
    return state.get("complexity", "medium")


# ─── Main graph ─────────────────────────────────────────────────────────────────

## Build and compile the top-level orchestration graph
def create_main_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled LangGraph graph ready for invocation"
    }
    """
    graph = StateGraph(GraphState)

    graph.add_node("discussion_panel", discussion_panel_node)
    graph.add_node("easy_tier",        easy_node)
    graph.add_node("medium_tier",      medium_node)
    graph.add_node("complex_tier",     complex_node)

    graph.add_edge(START, "discussion_panel")
    graph.add_conditional_edges(
        "discussion_panel",
        route_by_complexity,
        {
            "easy":    "easy_tier",
            "medium":  "medium_tier",
            "complex": "complex_tier",
        },
    )
    graph.add_edge("easy_tier",    END)
    graph.add_edge("medium_tier",  END)
    graph.add_edge("complex_tier", END)

    return graph.compile()
