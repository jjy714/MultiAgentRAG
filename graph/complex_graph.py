from langgraph.graph import START, END, StateGraph
from typing import Literal
from graph.state import GraphState, PlanExecState, RagState
from agents.PlannerAgent import PlannerAgent
from agents.StepDefinerAgent import StepDefinerAgent
from agents.retriever_node import retriever_node
from agents.ExtractorAgent import ExtractorAgent
from agents.QuestionAnsweringAgent import QuestionAnsweringAgent
from agents.WebSearchAgent import WebSearchAgent


def create_single_rag_graph():
    g = StateGraph(RagState)
    g.add_node("retrieve", retriever_node)
    g.add_node("extract", ExtractorAgent)
    g.add_node("answer", QuestionAnsweringAgent)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile()


def create_single_web_graph():
    g = StateGraph(RagState)
    g.add_node("web_search", WebSearchAgent)
    g.add_node("extract", ExtractorAgent)
    g.add_node("answer", QuestionAnsweringAgent)
    g.add_edge(START, "web_search")
    g.add_edge("web_search", "extract")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile()


def single_rag_execute_node(state: PlanExecState) -> dict:
    next_idx = len(state.get("step_output", []))
    result = create_single_rag_graph().invoke({
        "question": state["step_question"][next_idx],
        "token_usage": {}
    })
    raw = result.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    print(f"[complex] RAG step {next_idx}: {answer[:80]}...")
    return {
        "step_output": [{"task": state["step_question"][next_idx], "answer": answer}],
        "step_notes": result.get("notes", []),
        "token_usage": result.get("token_usage", {})
    }


async def single_web_execute_node(state: PlanExecState) -> dict:
    next_idx = len(state.get("step_output", []))
    result = await create_single_web_graph().ainvoke({
        "question": state["step_question"][next_idx],
        "token_usage": {}
    })
    raw = result.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    print(f"[complex] Web step {next_idx}: {answer[:80]}...")
    return {
        "step_output": [{"task": state["step_question"][next_idx], "answer": answer}],
        "step_notes": result.get("notes", []),
        "token_usage": result.get("token_usage", {})
    }


def _router_branch(state: PlanExecState) -> Literal["rag_execute", "web_execute", "__end__"]:
    current_idx = len(state.get("step_output", []))
    if current_idx >= len(state.get("plan", [])):
        return END
    step_type = state.get("step_question", [])[current_idx].get("type", "")
    if step_type == "retrieve_db":
        return "rag_execute"
    elif step_type == "web_search":
        return "web_execute"
    return END


def create_plan_executor_graph():
    g = StateGraph(PlanExecState)
    g.add_node("step_definer", StepDefinerAgent)
    g.add_node("rag_execute", single_rag_execute_node)
    g.add_node("web_execute", single_web_execute_node)

    g.set_entry_point("step_definer")
    g.add_conditional_edges(
        "step_definer",
        _router_branch,
        {"rag_execute": "rag_execute", "web_execute": "web_execute", END: END},
    )
    g.add_edge("rag_execute", "step_definer")
    g.add_edge("web_execute", "step_definer")
    return g.compile()


async def complex_executor_node(state: GraphState) -> dict:
    plan_exec_input: PlanExecState = {
        "original_question": state.get("original_question", ""),
        "plan": state.get("plan", []),
        "step_question": [],
        "step_output": [],
        "step_notes": [],
        "stop": False,
        "plan_summary": None,
        "token_usage": {},
    }
    result = await create_plan_executor_graph().ainvoke(plan_exec_input)
    return {
        "final_answer": result.get("plan_summary", ""),
        "token_usage": result.get("token_usage", {})
    }


async def web_augment_node(state: GraphState) -> dict:
    question = state.get("original_question", "")
    web_result = await WebSearchAgent({"question": question})
    web_docs = web_result.get("documents", [""])
    web_content = web_docs[0] if web_docs else ""
    if web_content:
        merged = f"{state.get('final_answer', '')}\n\n[Web Context]\n{web_content}".strip()
        print(f"[complex] Web augmentation added ({len(web_content)} chars).")
    else:
        merged = state.get("final_answer", "")
    return {
        "final_answer": merged,
        "web_result": web_content,
        "token_usage": web_result.get("token_usage", {})
    }


def _route_web(state: GraphState) -> str:
    return "web_augment" if state.get("web_needed", False) else END


def create_complex_graph():
    g = StateGraph(GraphState)
    g.add_node("planner", PlannerAgent)
    g.add_node("executor", complex_executor_node)
    g.add_node("web_augment", web_augment_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor",
        _route_web,
        {"web_augment": "web_augment", END: END},
    )
    g.add_edge("web_augment", END)
    return g.compile()
