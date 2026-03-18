from pydantic import BaseModel
from schema.Message import Message


### Pydantic model for an incoming chat request from the frontend
class ChatRequest(BaseModel):
    """
    args   : {
        "message (str)": "the user's current message text",
        "history (list[Message])": "list of prior conversation messages"
    }
    return : {
        "ChatRequest": "validated Pydantic model instance"
    }
    """
    message: str
    history: list[Message] = []
