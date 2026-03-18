import os
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from graph.state import PlanExecState, StepTaskState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/StepDefinerAgent.yaml"
SUMMARY_AGENT_PROMPT = "agents/Prompts/PlanSummarizerAgent.yaml"


## LangGraph node function that defines precise tasks for each plan step, or summarizes all results when done
def StepDefinerAgent(state: PlanExecState) -> dict:
    """
    args   : {
        "state (PlanExecState)": "plan execution state with 'plan', 'step_output', 'step_question', 'step_notes', and 'original_question'"
    }
    return : {
        "dict": "updated state — either 'step_question' (List[StepTaskState]) for the next step, or 'stop' (bool) and 'plan_summary' (str) when all steps are complete"
    }
    """
    plan = state.get("plan", [])
    step_output = state.get("step_output", [])
    step_question = state.get("step_question", [])
    step_notes = state.get("step_notes", [])
    original_question = state.get("original_question", "")
    finished_step_id = len(step_output)

    llm = get_llm()

    # All steps complete — produce final summary
    if finished_step_id >= len(plan):
        summary_system = get_system_prompt(SUMMARY_AGENT_PROMPT)
        summary_user = get_user_prompt(SUMMARY_AGENT_PROMPT)
        summary = llm.invoke([
            SystemMessage(summary_system),
            HumanMessage(summary_user.format(
                step_notes=step_notes,
                original_question=original_question
            ))
        ])
        print(f"[StepDefinerAgent] All {len(plan)} steps done. Final summary produced.")
        return {
            "stop": True,
            "plan_summary": summary.content,
        }

    # Define the next step
    cur_step = plan[finished_step_id]
    memory_str = "\n".join(
        f"Q: {q.get('task', q)}, A: {a.get('answer', a)}"
        for q, a in zip(step_question, step_output)
    )
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    structured_llm = llm.with_structured_output(StepTaskState)
    result: StepTaskState = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(plan=plan, cur_step=cur_step, memory=memory_str))
    ])
    print(f"[StepDefinerAgent] Step {finished_step_id}: task={result.get('task')} type={result.get('type')}")
    return {"step_question": [result]}
