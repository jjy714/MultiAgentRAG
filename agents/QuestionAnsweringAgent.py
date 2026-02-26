import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from graph.state import QAAnswerState

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_NAME = os.getenv("VLLM_NAME")
AGENT_PROMPT = "agents/Prompts/QuestionAnsweringAgent.yaml"


def QuestionAnsweringAgent(state: dict) -> dict:
    """Answers the question using retrieved + extracted context."""
    question = state.get("question", "")
    documents = state.get("documents", [])
    notes = state.get("notes", [])
    # Combine documents and extracted notes as context
    context = "\n\n".join(notes) if notes else "\n\n".join(str(d) for d in documents)

    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = ChatOpenAI(
        model_name=VLLM_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    structured_llm = llm.with_structured_output(QAAnswerState)
    result: QAAnswerState = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(context=context, question=question))
    ])
    print(f"[QuestionAnsweringAgent] confidence={result.get('confidence')}")
    return {"final_raw_answer": result}
