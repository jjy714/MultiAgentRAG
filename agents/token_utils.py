from __future__ import annotations


def normalize_token_usage(usage_metadata: dict) -> dict[str, int]:
    if not usage_metadata:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0))
    output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0))
    total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
