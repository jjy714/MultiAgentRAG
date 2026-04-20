import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import get_token_usage, strip_json_fence
from graph.state import QAAnswerState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/QuestionAnsweringAgent.yaml"


def QuestionAnsweringAgent(state: dict) -> dict:
    question = state.get("question", "")
    notes = state.get("notes", [])
    documents = state.get("documents", [])
    context = "\n\n".join(notes) if notes else "\n\n".join(str(d) for d in documents)

    response = get_llm().invoke([
        SystemMessage(get_system_prompt(AGENT_PROMPT)),
        HumanMessage(get_user_prompt(AGENT_PROMPT).format(context=context, question=question))
    ])

    try:
        result = QAAnswerState(**json.loads(strip_json_fence(response.content)))
    except Exception:
        result = QAAnswerState(answer=response.content, confidence="medium", sources=[])

    print(f"[QuestionAnsweringAgent] confidence={result.get('confidence')}")
    return {"final_raw_answer": result, "token_usage": get_token_usage(response)}
