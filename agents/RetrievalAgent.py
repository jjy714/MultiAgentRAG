from Agent import Agent
from tools import dense_search, sparse_search, hybrid_search
from langgraph.prebuilt import create_react_agent


class RetrievalAgent(Agent):
    
    def __init__(self, llm):
        self.agent = create_react_agent(model=llm, tools=[dense_search, sparse_search, hybrid_search])

    
    
    
    
    