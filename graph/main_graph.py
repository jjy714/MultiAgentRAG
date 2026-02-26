from langgraph.graph import START, END, StateGraph
from graph.state import GraphState, DiscussionState
from graph.discussion_graph import create_discussion_graph
from graph.easy_graph import create_easy_graph
from graph.medium_graph import create_medium_graph
from graph.complex_graph import create_complex_graph


# ─── Discussion Panel node ──────────────────────────────────────────────────────

async def discussion_panel_node(state: GraphState) -> dict:
    """
    Runs the discussion graph (3 parallel advocate agents + moderator).
    Writes the resulting 'complexity' back to GraphState.
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

async def easy_node(state: GraphState) -> dict:
    easy_graph = create_easy_graph()
    return await easy_graph.ainvoke(state)


async def medium_node(state: GraphState) -> dict:
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


async def complex_node(state: GraphState) -> dict:
    complex_graph = create_complex_graph()
    return await complex_graph.ainvoke(state)


# ─── Router ─────────────────────────────────────────────────────────────────────

def route_by_complexity(state: GraphState) -> str:
    return state.get("complexity", "medium")


# ─── Main graph ─────────────────────────────────────────────────────────────────

def create_main_graph():
    """
    Top-level graph:
      discussion_panel → [easy | medium | complex] → END
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
