import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Ollama / vLLM connection settings ─────────────────────────────────────────
# All agents share this single factory so the endpoint is configured once.
# Override the env-vars below to switch inference backends without touching code.

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1")
VLLM_NAME     = os.getenv("VLLM_NAME",     "qwen3.5:2b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")


## Return a ChatOpenAI instance pointed at the configured Ollama/vLLM endpoint
def get_llm(temperature: float = 0, **kwargs) -> ChatOpenAI:
    """
    args   : {
        "temperature (float)": "sampling temperature — 0 for deterministic output",
        "**kwargs": "extra parameters forwarded to ChatOpenAI"
    }
    return : {
        "ChatOpenAI": "LangChain ChatOpenAI client pointing at VLLM_BASE_URL"
    }
    """
    return ChatOpenAI(
        model=VLLM_NAME,
        base_url=VLLM_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        **kwargs,
    )
