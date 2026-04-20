from langgraph.graph import START, END, StateGraph
from graph.state import GraphState, DiscussionState
from graph.discussion_graph import create_discussion_graph
from graph.easy_graph import create_easy_graph
from graph.medium_graph import create_medium_graph
from graph.complex_graph import create_complex_graph


async def discussion_panel_node(state: GraphState) -> dict:
    discussion_graph = create_discussion_graph()
    discussion_input: DiscussionState = {
        "question": state.get("original_question", ""),
        "votes": [],
        "complexity": "",
        "token_usage": {},
    }
    result = await discussion_graph.ainvoke(discussion_input)
    complexity = result.get("complexity", "medium")
    token_usage = result.get("token_usage", {})
    print(f"\n{'='*50}")
    print(f"[DiscussionPanel] Routing to tier: '{complexity}'")
    print(f"{'='*50}\n")
    return {"complexity": complexity, "token_usage": token_usage}


async def easy_node(state: GraphState) -> dict:
    easy_graph = create_easy_graph()
    result = await easy_graph.ainvoke(state)
    return {
        "final_answer": result.get("final_answer"),
        "token_usage": result.get("token_usage", {}),
    }


async def medium_node(state: GraphState) -> dict:
    medium_graph = create_medium_graph()
    # The medium subgraph uses RagState, so we project the relevant fields
    # rather than passing GraphState directly.
    medium_input = {
        "question": state.get("original_question", ""),
        "documents": [],
        "doc_ids": [],
        "notes": [],
        "final_raw_answer": None,
        "token_usage": {},
    }
    result = await medium_graph.ainvoke(medium_input)
    raw = result.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    return {
        "final_answer": answer,
        "token_usage": result.get("token_usage", {}),
    }


async def complex_node(state: GraphState) -> dict:
    complex_graph = create_complex_graph()
    result = await complex_graph.ainvoke(state)
    return {
        "final_answer": result.get("final_answer"),
        "token_usage": result.get("token_usage", {}),
    }


def route_by_complexity(state: GraphState) -> str:
    return state.get("complexity", "medium")


def create_main_graph():
    graph = StateGraph(GraphState)

    graph.add_node("discussion_panel", discussion_panel_node)
    graph.add_node("easy_tier", easy_node)
    graph.add_node("medium_tier", medium_node)
    graph.add_node("complex_tier", complex_node)

    graph.add_edge(START, "discussion_panel")
    graph.add_conditional_edges(
        "discussion_panel",
        route_by_complexity,
        {
            "easy": "easy_tier",
            "medium": "medium_tier",
            "complex": "complex_tier",
        },
    )
    graph.add_edge("easy_tier", END)
    graph.add_edge("medium_tier", END)
    graph.add_edge("complex_tier", END)

    return graph.compile()
