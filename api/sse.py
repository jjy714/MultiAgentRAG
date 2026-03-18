import json

## Serialize a payload dict into an SSE-formatted data string
def sse(payload: dict) -> str:
    """
    args   : {
        "payload (dict)": "event data to serialize as JSON"
    }
    return : {
        "str": "SSE-formatted string with 'data: ' prefix and double newline"
    }
    """
    return f"data: {json.dumps(payload)}\n\n"