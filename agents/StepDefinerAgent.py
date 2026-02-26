import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from graph.state import PlanExecState, StepTaskState

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_NAME = os.getenv("VLLM_NAME")
AGENT_PROMPT = "agents/Prompts/StepDefinerAgent.yaml"
SUMMARY_AGENT_PROMPT = "agents/Prompts/PlanSummarizerAgent.yaml"


def StepDefinerAgent(state: PlanExecState) -> dict:
    """
    For each step in the plan, defines the precise task and its execution type.
    When all steps are done, summarizes all notes into a final answer.
    """
    plan = state.get("plan", [])
    step_output = state.get("step_output", [])
    step_question = state.get("step_question", [])
    step_notes = state.get("step_notes", [])
    original_question = state.get("original_question", "")
    finished_step_id = len(step_output)

    llm = ChatOpenAI(
        model_name=VLLM_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

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
