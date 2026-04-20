import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import get_token_usage, strip_json_fence
from graph.state import PlanExecState, StepTaskState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/StepDefinerAgent.yaml"
SUMMARY_AGENT_PROMPT = "agents/Prompts/PlanSummarizerAgent.yaml"


def StepDefinerAgent(state: PlanExecState) -> dict:
    plan = state.get("plan", [])
    step_output = state.get("step_output", [])
    step_question = state.get("step_question", [])
    step_notes = state.get("step_notes", [])
    original_question = state.get("original_question", "")
    current_step = len(step_output)

    llm = get_llm()

    if current_step >= len(plan):
        response = llm.invoke([
            SystemMessage(get_system_prompt(SUMMARY_AGENT_PROMPT)),
            HumanMessage(get_user_prompt(SUMMARY_AGENT_PROMPT).format(
                step_notes=step_notes,
                original_question=original_question
            ))
        ])
        print(f"[StepDefinerAgent] All {len(plan)} steps done. Final summary produced.")
        return {
            "stop": True,
            "plan_summary": response.content,
            "token_usage": get_token_usage(response)
        }

    cur_step = plan[current_step]
    # Build a compact memory string from completed steps so the LLM can frame
    # the next task in the context of what has already been answered.
    memory_str = "\n".join(
        f"Q: {q.get('task', q) if isinstance(q, dict) else q}, A: {a.get('answer', a) if isinstance(a, dict) else a}"
        for q, a in zip(step_question, step_output)
    )
    response = llm.invoke([
        SystemMessage(get_system_prompt(AGENT_PROMPT)),
        HumanMessage(get_user_prompt(AGENT_PROMPT).format(plan=plan, cur_step=cur_step, memory=memory_str))
    ])

    try:
        task = StepTaskState(**json.loads(strip_json_fence(response.content)))
    except Exception:
        task = StepTaskState(task=cur_step, type="retrieve_db")

    print(f"[StepDefinerAgent] Step {current_step}: task={task.get('task')} type={task.get('type')}")
    return {"step_question": [task], "token_usage": get_token_usage(response)}
