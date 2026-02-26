from typing import TypedDict


class StepTaskState(TypedDict):
    task: str   # the sub-question / task description for this step
    type: str   # "retrieve_db" | "web_search"
