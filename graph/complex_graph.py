from langgraph.graph import START, END, StateGraph
from typing import Literal
from graph.state import GraphState, PlanExecState, RagState
from agents.PlannerAgent import PlannerAgent
from agents.StepDefinerAgent import StepDefinerAgent
from agents.retriever_node import retriever_node
from agents.ExtractorAgent import ExtractorAgent
from agents.QuestionAnsweringAgent import QuestionAnsweringAgent
from agents.WebSearchAgent import WebSearchAgent


# ─── Inner single-task graphs ──────────────────────────────────────────────────

def create_single_rag_graph():
    """retrieve → extract → answer (for one sub-question via DB)."""
    g = StateGraph(RagState)
    g.add_node("retrieve", retriever_node)
    g.add_node("extract",  ExtractorAgent)
    g.add_node("answer",   QuestionAnsweringAgent)
    g.add_edge(START,      "retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract",  "answer")
    g.add_edge("answer",   END)
    return g.compile()


async def create_single_web_graph():
    """web_search → extract → answer (for one sub-question via web)."""
    g = StateGraph(RagState)
    g.add_node("web_search", WebSearchAgent)
    g.add_node("extract",    ExtractorAgent)
    g.add_node("answer",     QuestionAnsweringAgent)
    g.add_edge(START,        "web_search")
    g.add_edge("web_search", "extract")
    g.add_edge("extract",    "answer")
    g.add_edge("answer",     END)
    return g.compile()


# ─── Plan-executor inner loop nodes ────────────────────────────────────────────

def single_rag_execute_node(state: PlanExecState) -> dict:
    next_idx = len(state.get("step_output", []))
    rag_graph = create_single_rag_graph()
    result = rag_graph.invoke({"question": state["step_question"][next_idx]})
    raw = result.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    print(f"[complex] RAG step {next_idx}: {answer[:80]}...")
    return {
        "step_output": [{"task": state["step_question"][next_idx], "answer": answer}],
        "step_notes": result.get("notes", []),
    }


async def single_web_execute_node(state: PlanExecState) -> dict:
    next_idx = len(state.get("step_output", []))
    web_graph = await create_single_web_graph()
    result = await web_graph.ainvoke({"question": state["step_question"][next_idx]})
    raw = result.get("final_raw_answer", {})
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)
    print(f"[complex] Web step {next_idx}: {answer[:80]}...")
    return {
        "step_output": [{"task": state["step_question"][next_idx], "answer": answer}],
        "step_notes": result.get("notes", []),
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


# ─── Plan-executor loop graph ───────────────────────────────────────────────────

def create_plan_executor_graph():
    """
    StepDefinerAgent → [retrieve_db | web_search] → loops back → StepDefinerAgent
    until all steps are done (stop=True), then exits to END.
    """
    g = StateGraph(PlanExecState)
    g.add_node("step_definer", StepDefinerAgent)
    g.add_node("rag_execute",  single_rag_execute_node)
    g.add_node("web_execute",  single_web_execute_node)

    g.set_entry_point("step_definer")
    g.add_conditional_edges(
        "step_definer",
        _router_branch,
        {"rag_execute": "rag_execute", "web_execute": "web_execute", END: END},
    )
    g.add_edge("rag_execute", "step_definer")
    g.add_edge("web_execute", "step_definer")
    return g.compile()


# ─── Complex tier top-level ─────────────────────────────────────────────────────

async def complex_executor_node(state: GraphState) -> dict:
    """Invokes the plan-executor loop. Returns plan_summary as final_answer."""
    plan_executor = create_plan_executor_graph()
    plan_exec_input: PlanExecState = {
        "original_question": state.get("original_question", ""),
        "plan": state.get("plan", []),
        "step_question": [],
        "step_output":   [],
        "step_notes":    [],
        "stop": False,
        "plan_summary": None,
    }
    result = await plan_executor.ainvoke(plan_exec_input)
    return {"final_answer": result.get("plan_summary", "")}


def create_complex_graph():
    """
    Complex tier:
      PlannerAgent → complex_executor_node (StepDefiner loop) → END
    """
    g = StateGraph(GraphState)
    g.add_node("planner",  PlannerAgent)
    g.add_node("executor", complex_executor_node)
    g.add_edge(START,      "planner")
    g.add_edge("planner",  "executor")
    g.add_edge("executor", END)
    return g.compile()
