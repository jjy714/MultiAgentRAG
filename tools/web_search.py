from langchain_core.tools import tool
from langgraph.types import Command
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()

# API key for the Google Custom Search service
GOOGLE_CUSTOM_SEARCH_API = os.getenv("GOOGLE_CUSTOM_SEARCH_API")

## LangChain tool that returns a configured GoogleSearchAPIWrapper for real-time web search
@tool
def web_search_tool():
    """
    args   : {}
    return : {
        "GoogleSearchAPIWrapper": "configured Google search wrapper returning top 3 results"
    }
    """
    return GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_CUSTOM_SEARCH_API,
        k=3
    )