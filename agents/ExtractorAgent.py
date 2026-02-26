import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
AGENT_PROMPT = "agents/Prompts/ExtractorAgent.yaml"


def ExtractorAgent(state: dict) -> dict:
    """Extracts relevant information from retrieved documents for the current query."""
    passage = state.get("documents", [])
    question = state.get("question", "")
    notes = state.get("notes", [])

    system_prompt = get_system_prompt(AGENT_PROMPT)
    user_prompt = get_user_prompt(AGENT_PROMPT)

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    result = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(passage=passage, query=question))
    ])
    notes.append(result.content)
    print(f"[ExtractorAgent] Extracted {len(result.content)} chars of notes.")
    return {"notes": notes}
