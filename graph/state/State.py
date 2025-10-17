from langgraph.graph import MessagesState
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from graph.schema.section import Section
from typing import TypedDict, Annotated
import operator
from abc import ABC


class State(TypedDict):

    pass



class AgentState(MessagesState):
    
    def __init__(self):
        super().__init__()
        



# Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report


# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]
