from langgraph.graph import MessagesState
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from graph.schema.section import Section
from typing import TypedDict, Annotated
import operator
from abc import ABC


### Placeholder TypedDict state (base / unused — reserved for future use)
class State(TypedDict):
    """
    args   : {}
    return : {
        "State": "empty typed dict"
    }
    """
    pass


### Extended MessagesState for multi-agent graph execution with message history support
class AgentState(MessagesState):
    """
    args   : {}
    return : {
        "AgentState": "agent state with full message history"
    }
    """

    ## Initialize AgentState by delegating to the MessagesState parent constructor
    def __init__(self):
        """
        args   : {}
        return : {
            "None": "delegates to MessagesState.__init__"
        }
        """
        super().__init__()


### Graph state for report-generation workflows with section planning support
class State(TypedDict):
    """
    args   : {
        "topic (str)": "the report topic",
        "sections (list[Section])": "planned sections for the report",
        "completed_sections (list)": "sections written by parallel workers",
        "final_report (str)": "the assembled final report text"
    }
    return : {
        "State": "typed dict for report-generation graph state"
    }
    """
    topic: str              # Report topic
    sections: list[Section] # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]                       # All workers write to this key in parallel
    final_report: str       # Final report


### State for a single report-section worker node
class WorkerState(TypedDict):
    """
    args   : {
        "section (Section)": "the specific section this worker is responsible for",
        "completed_sections (list)": "accumulator for completed section outputs"
    }
    return : {
        "WorkerState": "typed dict for a report section worker"
    }
    """
    section: Section
    completed_sections: Annotated[list, operator.add]
