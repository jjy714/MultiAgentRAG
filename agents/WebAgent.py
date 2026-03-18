from Agent import Agent
from langgraph.prebuilt import create_react_agent
from tools import web_search_tool


### Agent class that performs real-time web searches using a React-based tool-calling pattern
class WebAgent(Agent):
    """
    args   : {
        "llm": "the underlying language model used by the ReAct agent"
    }
    return : {
        "WebAgent": "agent instance with a ReAct executor bound to the web search tool"
    }
    """

    ## Initialize the agent with a ReAct executor using the web search tool
    def __init__(self, llm):
        """
        args   : {
            "llm": "language model compatible with create_react_agent"
        }
        return : {
            "None": "initializes self.agent"
        }
        """
        self.agent = create_react_agent(model=llm, tools=[web_search_tool])

    ## Delegate communication to the parent Agent base class
    def communicate(self):
        """
        args   : {}
        return : {
            "Any": "result from the parent Agent communicate implementation"
        }
        """
        return super().communicate()
