import os
import re
from urllib.parse import urlencode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from agents.llm import get_llm

load_dotenv()

# Environment variable configuration for the Exa MCP web search service
SMITHERY_EXA_URL = os.getenv("SMITHERY_EXA_URL")
SMITHERY_EXA_PARAMS = {
    "api_key": os.getenv("EXA_MCP_API_KEY"),
    "profile": os.getenv("EXA_MCP_PROFILE"),
}

# Regex pattern to strip chain-of-thought <think> tags from LLM output
THINK_TAG_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")


## LangGraph node function that performs a real-time web search via Exa MCP
async def WebSearchAgent(state: dict) -> dict:
    """
    args   : {
        "state (dict)": "graph state with 'question' (str) or 'original_question' (str)"
    }
    return : {
        "dict": "updated state with 'documents' (List[str]) containing the web search result"
    }
    """
    question = state.get("question", "") or state.get("original_question", "")
    if isinstance(question, dict):
        # Extract plain-text task string if question is a structured dict
        question = question.get("task", "")

    client = MultiServerMCPClient(
        {
            "exa": {
                "transport": "streamable_http",
                "url": f"{SMITHERY_EXA_URL}?{urlencode(SMITHERY_EXA_PARAMS)}",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        model=get_llm(),
        tools=tools,
    )
    response = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    messages = response.get("messages", [])

    # Extract the last AIMessage content, stripping chain-of-thought tags
    final_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_content = THINK_TAG_RE.sub("", msg.content).strip()
            break

    print(f"[WebSearchAgent] Retrieved {len(final_content)} chars from web.")
    return {"documents": [final_content]}
