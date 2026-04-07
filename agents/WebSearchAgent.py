import os
import re
import httpx
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from agents.llm import get_llm
from agents.token_utils import normalize_token_usage
from graph.state.GraphState import sum_dicts

load_dotenv()

EXA_SEARCH_API_KEY = os.getenv("EXA_SEARCH_API_KEY")
EXA_API_URL = "https://api.exa.ai/search"

# Regex pattern to strip chain-of-thought <think> tags from LLM output
THINK_TAG_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")


## LangGraph node function that performs a real-time web search via Exa REST API
async def WebSearchAgent(state: dict) -> dict:
    """
    args   : {
        "state (dict)": "graph state with 'question' (str) or 'original_question' (str)"
    }
    return : {
        "dict": "updated state with 'documents' (List[str]) and 'token_usage'"
    }
    """
    question = state.get("question", "") or state.get("original_question", "")
    if isinstance(question, dict):
        question = question.get("task", "")

    if not EXA_SEARCH_API_KEY:
        print("[WebSearchAgent] EXA_SEARCH_API_KEY not configured — skipping web search.")
        return {"documents": [], "token_usage": {}}

    # Retrieve search results from Exa REST API
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                EXA_API_URL,
                headers={
                    "x-api-key": EXA_SEARCH_API_KEY,
                    "Content-Type": "application/json",
                },
                json={"query": question, "numResults": 5, "contents": {"text": True}},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        print(f"[WebSearchAgent] Exa API call failed: {exc}")
        return {"documents": [], "token_usage": {}}

    MAX_SNIPPET_CHARS = 2000
    snippets = [r.get("text", "")[:MAX_SNIPPET_CHARS] for r in data.get("results", []) if r.get("text")]
    if not snippets:
        print("[WebSearchAgent] Exa returned no text results.")
        return {"documents": [], "token_usage": {}}

    context = "\n\n".join(snippets)
    print(f"[WebSearchAgent] Retrieved {len(snippets)} results ({len(context)} chars) from Exa.")

    # Synthesize a concise answer using the active LLM backend
    synthesis_prompt = (
        f"Using the following web search results, provide a concise and accurate answer "
        f"to the question.\n\nQuestion: {question}\n\nSearch Results:\n{context}"
    )
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(synthesis_prompt)])

    final_content = THINK_TAG_RE.sub("", response.content).strip()

    token_usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage = normalize_token_usage(response.usage_metadata)

    print(f"[WebSearchAgent] Synthesized {len(final_content)} chars from web.")
    return {
        "documents": [final_content],
        "token_usage": token_usage,
    }
