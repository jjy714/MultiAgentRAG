from typing import TypedDict


### Structured output schema for a single step task defined by StepDefinerAgent
class StepTaskState(TypedDict):
    """
    args   : {
        "task (str)": "the sub-question or task description for this step",
        "type (str)": "execution method — 'retrieve_db' or 'web_search'"
    }
    return : {
        "StepTaskState": "typed dict defining the task and its retrieval strategy"
    }
    """
    task: str   # the sub-question / task description for this step
    type: str   # "retrieve_db" | "web_search"
