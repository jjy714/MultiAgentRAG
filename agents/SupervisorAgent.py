from Agent import Agent
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.prebuilt import InjectedState, create_react_agent
from tools import transfer_tool


class SupervisorAgent(Agent):
    
    def __init__(self, tools: list):
        self.tools = tools
    
    def agent_1(state: Annotated[dict, InjectedState]):
        # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
        # and add any additional logic (different models, custom prompts, structured output, etc.)
        response = model.invoke(...)
        # return the LLM response as a string (expected tool response format)
        # this will be automatically turned to ToolMessage
        # by the prebuilt create_react_agent (supervisor)
        return response.content
    


model = ChatOpenAI()

# this is the agent function that will be called as tool
# notice that you can pass the state to the tool via InjectedState annotation


def agent_2(state: Annotated[dict, InjectedState]):
    response = model.invoke(...)
    return response.content

# the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
# that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
supervisor = create_react_agent(model, tools)

