from pydantic import BaseModel
from schema.Message import Message


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
