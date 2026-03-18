import os
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from graph.state import QAAnswerState

load_dotenv()

AGENT_PROMPT = "agents/Prompts/QuestionAnsweringAgent.yaml"


## LangGraph node function that synthesizes a final answer from retrieved context and extracted notes
def QuestionAnsweringAgent(state: dict) -> dict:
    """
    args   : {
        "state (dict)": "graph state with 'question' (str), 'documents' (List[str]), 'notes' (List[str])"
    }
    return : {
        "dict": "updated state with 'final_raw_answer' (QAAnswerState)"
    }
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    notes = state.get("notes", [])
    # Combine documents and extracted notes as context
    context = "\n\n".join(notes) if notes else "\n\n".join(str(d) for d in documents)

    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    structured_llm = get_llm().with_structured_output(QAAnswerState)
    result: QAAnswerState = structured_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(context=context, question=question))
    ])
    print(f"[QuestionAnsweringAgent] confidence={result.get('confidence')}")
    return {"final_raw_answer": result}
