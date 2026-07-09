from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from agents.load_prompt import get_system_prompt, get_user_prompt
from agents.llm import get_llm
from agents.token_utils import get_token_usage

load_dotenv()

AGENT_PROMPT = "agents/Prompts/ExtractorAgent.yaml"


def ExtractorAgent(state: dict) -> dict:
    passage = state.get("documents", [])
    question = state.get("question", "")
    print(f"[ExtractorAgent] Reference documents ({len(passage)}):")
    for i, doc in enumerate(passage):
        print(f"  [{i+1}] {' '.join(doc.split()[:50])}")
    notes = state.get("notes", [])

    response = get_llm().invoke([
        SystemMessage(get_system_prompt(AGENT_PROMPT)),
        HumanMessage(get_user_prompt(AGENT_PROMPT).format(passage=passage, query=question))
    ])

    notes.append(response.content)
    print(f"[ExtractorAgent] Extracted {len(response.content)} chars of notes.")
    return {"notes": notes, "token_usage": get_token_usage(response)}
