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


## Build and compile a single-pass RAG subgraph for one sub-question via database retrieval
def create_single_rag_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled retrieve → extract → answer graph"
    }
    """
    g = StateGraph(RagState)
    g.add_node("retrieve", retriever_node)
    g.add_node("extract", ExtractorAgent)
    g.add_node("answer", QuestionAnsweringAgent)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile()


## Build and compile a single-pass web search subgraph for one sub-question via web
async def create_single_web_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled web_search → extract → answer graph"
    }
    """
    g = StateGraph(RagState)
    g.add_node("web_search", WebSearchAgent)
    g.add_node("extract", ExtractorAgent)
    g.add_node("answer", QuestionAnsweringAgent)
    g.add_edge(START, "web_search")
    g.add_edge("web_search", "extract")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile()


# ─── Plan-executor inner loop nodes ────────────────────────────────────────────


## Execute one database-retrieval step from the plan and append its output to state
def single_rag_execute_node(state: PlanExecState) -> dict:
    """
    args   : {
        "state (PlanExecState)": "plan execution state with 'step_question' and 'step_output'"
    }
    return : {
        "dict": "updated state appending 'step_output' (dict), 'step_notes' (List[str]), and 'token_usage'"
    }
    """
    next_idx = len(state.get("step_output", []))
    rag_graph = create_single_rag_graph()
    result = rag_graph.invoke({
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


## Execute one web-search step from the plan and append its output to state
async def single_web_execute_node(state: PlanExecState) -> dict:
    """
    args   : {
        "state (PlanExecState)": "plan execution state with 'step_question' and 'step_output'"
    }
    return : {
        "dict": "updated state appending 'step_output' (dict), 'step_notes' (List[str]), and 'token_usage'"
    }
    """
    next_idx = len(state.get("step_output", []))
    web_graph = await create_single_web_graph()
    result = await web_graph.ainvoke({
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


## Conditional edge function that decides whether to run RAG, web search, or terminate the loop
def _router_branch(
    state: PlanExecState,
) -> Literal["rag_execute", "web_execute", "__end__"]:
    """
    args   : {
        "state (PlanExecState)": "plan execution state with 'step_output' and 'plan'"
    }
    return : {
        "str": "'rag_execute', 'web_execute', or END depending on the next step type"
    }
    """
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


## Build and compile the plan-executor loop: StepDefiner → [RAG | Web] → loops back
def create_plan_executor_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled plan-executor graph that loops until all steps are done"
    }
    """
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


# ─── Complex tier top-level ─────────────────────────────────────────────────────


## LangGraph node that invokes the plan-executor loop and returns the plan summary as the final answer
async def complex_executor_node(state: GraphState) -> dict:
    """
    args   : {
        "state (GraphState)": "graph state with 'original_question' and 'plan'"
    }
    return : {
        "dict": "updated state with 'final_answer' (str) and 'token_usage' (dict)"
    }
    """
    plan_executor = create_plan_executor_graph()
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
    result = await plan_executor.ainvoke(plan_exec_input)
    return {
        "final_answer": result.get("plan_summary", ""),
        "token_usage": result.get("token_usage", {})
    }


## Build and compile the complex-tier graph: PlannerAgent → complex_executor_node
def create_complex_graph():
    """
    args   : {}
    return : {
        "CompiledStateGraph": "compiled complex-tier graph ready for invocation"
    }
    """
    g = StateGraph(GraphState)
    g.add_node("planner", PlannerAgent)
    g.add_node("executor", complex_executor_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", END)
    return g.compile()
