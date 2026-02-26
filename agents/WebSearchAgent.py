import os
import re
from urllib.parse import urlencode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

load_dotenv()

SMITHERY_EXA_URL = os.getenv("SMITHERY_EXA_URL")
SMITHERY_EXA_PARAMS = {
    "api_key": os.getenv("EXA_MCP_API_KEY"),
    "profile": os.getenv("EXA_MCP_PROFILE"),
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_NAME = os.getenv("VLLM_NAME")

THINK_TAG_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>")


async def WebSearchAgent(state: dict) -> dict:
    """Performs a real-time web search using Exa via MCP and returns the result as documents."""
    question = state.get("question", "") or state.get("original_question", "")
    if isinstance(question, dict):
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
        model=ChatOpenAI(
            model_name=VLLM_NAME,
            api_key=OPENAI_API_KEY,
        ),
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
