from Agent import Agent
from langgraph.prebuilt import create_react_agent



class SemanticAnalyzerAgent(Agent):
    
    def __init__(self, llm):
        self.agent = create_react_agent(model=llm, tools=[dense_search, sparse_search, hybrid_search])