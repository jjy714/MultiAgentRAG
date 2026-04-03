import os
from pathlib import Path
from dotenv import load_dotenv
from logger_tools import get_logger

current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / ".env.dev"
load_dotenv(env_path)

logger = get_logger(__name__)

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")


def get_llm(temperature: float = 0, **kwargs):
    if LLM_BACKEND == "local":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1")
        model = os.getenv("VLLM_NAME", "qwen2.5:3b")
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        logger.info(f"get_llm [local] model={model} base_url={base_url}")
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **kwargs,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        api_key = os.getenv("GOOGLE_API_KEY", "your_api_key_here")
        logger.info(f"get_llm [gemini] model={model}")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_retries=5,
            **kwargs,
        )
