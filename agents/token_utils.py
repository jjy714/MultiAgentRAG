from __future__ import annotations


def _to_int(v) -> int:
    """Safely coerce a value to int, returning 0 for non-numeric types.

    Newer LangChain versions occasionally return nested dicts inside
    usage_metadata instead of plain integers, so we guard against that here.
    """
    if isinstance(v, (int, float)):
        return int(v)
    return 0


def normalize_token_usage(usage_metadata: dict) -> dict[str, int]:
    if not usage_metadata:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens  = _to_int(usage_metadata.get("input_tokens",  usage_metadata.get("prompt_tokens", 0)))
    output_tokens = _to_int(usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0)))
    total_tokens  = _to_int(usage_metadata.get("total_tokens",  input_tokens + output_tokens))
    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total_tokens,
    }


def get_token_usage(response) -> dict[str, int]:
    meta = getattr(response, "usage_metadata", None) or {}
    return normalize_token_usage(meta)


def strip_json_fence(text: str) -> str:
    """Remove Markdown code fences from an LLM response before JSON parsing.

    Many models wrap their JSON output in ```json ... ``` blocks even when
    instructed not to, so this handles both the fenced and plain cases.
    """
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()
