from __future__ import annotations


def _to_int(v) -> int:
    """Safely convert a value to int, returning 0 for dicts or non-numeric types
    (newer LangChain versions may return nested dicts in usage_metadata)."""
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
