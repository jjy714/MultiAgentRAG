from langchain_core.tools import tool
from langgraph.types import Command
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_CUSTOM_SEARCH_API = os.getenv("GOOGLE_CUSTOM_SEARCH_API")

@tool
def web_search_tool():
    return GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_CUSTOM_SEARCH_API,
        k=3
    )