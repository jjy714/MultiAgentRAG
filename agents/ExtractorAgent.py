import os
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm

load_dotenv()

AGENT_PROMPT = "agents/Prompts/ExtractorAgent.yaml"


## LangGraph node function that extracts relevant information from retrieved documents
def ExtractorAgent(state: dict) -> dict:
    """
    args   : {
        "state (dict)": "graph state with 'documents' (List[str]), 'question' (str), 'notes' (List[str])"
    }
    return : {
        "dict": "updated state with 'notes' (List[str]) appended with extracted content"
    }
    """
    passage = state.get("documents", [])
    question = state.get("question", "")
    notes = state.get("notes", [])

    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = get_llm()
    result = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(passage=passage, query=question))
    ])
    notes.append(result.content)
    print(f"[ExtractorAgent] Extracted {len(result.content)} chars of notes.")
    return {"notes": notes}
