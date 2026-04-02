import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
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
        "dict": "updated state with 'final_raw_answer' (QAAnswerState) and 'token_usage'"
    }
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    notes = state.get("notes", [])
    # Combine documents and extracted notes as context
    context = "\n\n".join(notes) if notes else "\n\n".join(str(d) for d in documents)

    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(context=context, question=question))
    ])
    
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
        result = QAAnswerState(**data)
    except Exception:
        result = QAAnswerState(answer=response.content, confidence="medium", sources=[])

    token_usage = normalize_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else {})

    print(f"[QuestionAnsweringAgent] confidence={result.get('confidence')}")
    return {
        "final_raw_answer": result,
        "token_usage": token_usage
    }
