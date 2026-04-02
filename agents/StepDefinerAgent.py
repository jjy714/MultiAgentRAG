import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
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
        summary_response = llm.invoke([
            SystemMessage(summary_system),
            HumanMessage(summary_user.format(
                step_notes=step_notes,
                original_question=original_question
            ))
        ])
        token_usage = normalize_token_usage(summary_response.usage_metadata if hasattr(summary_response, 'usage_metadata') else {})
        print(f"[StepDefinerAgent] All {len(plan)} steps done. Final summary produced.")
        return {
            "stop": True,
            "plan_summary": summary_response.content,
            "token_usage": token_usage
        }

    # Define the next step
    cur_step = plan[finished_step_id]
    memory_str = "\n".join(
        f"Q: {q.get('task', q) if isinstance(q, dict) else q}, A: {a.get('answer', a) if isinstance(a, dict) else a}"
        for q, a in zip(step_question, step_output)
    )
    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(plan=plan, cur_step=cur_step, memory=memory_str))
    ])
    
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
        result = StepTaskState(**data)
    except Exception:
        result = StepTaskState(task=cur_step, type="retrieve_db")

    token_usage = normalize_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else {})

    print(f"[StepDefinerAgent] Step {finished_step_id}: task={result.get('task')} type={result.get('type')}")
    return {
        "step_question": [result],
        "token_usage": token_usage
    }
