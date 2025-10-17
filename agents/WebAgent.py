from Agent import Agent
from langgraph.prebuilt import create_react_agent
from tools import web_search_tool


class WebAgent(Agent):
    
    def __init__(self, llm):
        self.agent = create_react_agent(model=llm, tools=[web_search_tool])
    
    

    def communicate(self):
        return super().communicate() 
