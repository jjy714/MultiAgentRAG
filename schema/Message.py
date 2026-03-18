from pydantic import BaseModel


### Pydantic model for an individual chat message with a role and content
class Message(BaseModel):
    """
    args   : {
        "role (str)": "sender identity — 'user' or 'assistant'",
        "content (str)": "the message text"
    }
    return : {
        "Message": "validated Pydantic model instance"
    }
    """
    role: str    # "user" | "assistant"
    content: str