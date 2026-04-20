from typing import TypedDict


class StepTaskState(TypedDict):
    task: str
    type: str  # "retrieve_db" | "web_search"
