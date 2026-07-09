import os
import re
import httpx
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from agents.llm import get_llm
from agents.token_utils import get_token_usage

load_dotenv('../.env.dev')

EXA_SEARCH_API_KEY = os.getenv("EXA_SEARCH_API_KEY")
EXA_API_URL = "https://api.exa.ai/search"
THINK_TAG_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")
MAX_SNIPPET_CHARS = 2000


async def WebSearchAgent(state: dict) -> dict:
    question = state.get("question", "") or state.get("original_question", "")
    if isinstance(question, dict):
        question = question.get("task", "")

    if not EXA_SEARCH_API_KEY:
        print("[WebSearchAgent] EXA_SEARCH_API_KEY not configured — skipping web search.")
        return {"documents": [], "token_usage": {}}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                EXA_API_URL,
                headers={"x-api-key": EXA_SEARCH_API_KEY, "Content-Type": "application/json"},
                json={"query": question, "numResults": 5, "contents": {"text": True}},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        print(f"[WebSearchAgent] Exa API call failed: {exc}")
        return {"documents": [], "token_usage": {}}

    snippets = [r.get("text", "")[:MAX_SNIPPET_CHARS] for r in data.get("results", []) if r.get("text")]
    if not snippets:
        print("[WebSearchAgent] Exa returned no text results.")
        return {"documents": [], "token_usage": {}}

    context = "\n\n".join(snippets)
    print(f"[WebSearchAgent] Retrieved {len(snippets)} results ({len(context)} chars) from Exa.")

    prompt = (
        f"Using the following web search results, provide a concise and accurate answer "
        f"to the question.\n\nQuestion: {question}\n\nSearch Results:\n{context}"
    )
    response = await get_llm().ainvoke([HumanMessage(prompt)])
    answer = THINK_TAG_RE.sub("", response.content).strip()

    print(f"[WebSearchAgent] Synthesized {len(answer)} chars from web.")
    return {"documents": [answer], "token_usage": get_token_usage(response)}
