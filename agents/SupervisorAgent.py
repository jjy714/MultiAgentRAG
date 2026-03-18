from Agent import Agent
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.prebuilt import InjectedState, create_react_agent
from tools import transfer_tool


### Agent class that orchestrates other agents using a tool-calling ReAct supervisor pattern
class SupervisorAgent(Agent):
    """
    args   : {
        "tools (list)": "list of callable agent tools the supervisor can dispatch to"
    }
    return : {
        "SupervisorAgent": "agent instance configured to delegate to sub-agents via tools"
    }
    """

    ## Initialize the supervisor with a list of delegatable agent tools
    def __init__(self, tools: list):
        """
        args   : {
            "tools (list)": "list of callable agent-as-tool functions"
        }
        return : {
            "None": "initializes self.tools"
        }
        """
        self.tools = tools

    ## Sub-agent function 1 — processes injected state and returns an LLM response
    def agent_1(state: Annotated[dict, InjectedState]):
        """
        args   : {
            "state (Annotated[dict, InjectedState])": "current graph state passed by the supervisor"
        }
        return : {
            "str": "LLM response content to be returned as a ToolMessage"
        }
        """
        # Pass relevant parts of the state to the LLM (e.g., state["messages"])
        response = model.invoke(...)
        # Return the LLM response as a string; automatically wrapped as ToolMessage
        return response.content


model = ChatOpenAI()

# Sub-agent function 2 — processes injected state and returns an LLM response
def agent_2(state: Annotated[dict, InjectedState]):
    """
    args   : {
        "state (Annotated[dict, InjectedState])": "current graph state passed by the supervisor"
    }
    return : {
        "str": "LLM response content to be returned as a ToolMessage"
    }
    """
    response = model.invoke(...)
    return response.content

# Build a supervisor using the prebuilt ReAct agent with tool-calling
supervisor = create_react_agent(model, tools)
