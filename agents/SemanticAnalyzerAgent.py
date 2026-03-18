from Agent import Agent
from langgraph.prebuilt import create_react_agent


### Agent class that performs semantic analysis using dense, sparse, and hybrid search tools
class SemanticAnalyzerAgent(Agent):
    """
    args   : {
        "llm": "the underlying language model used by the ReAct agent"
    }
    return : {
        "SemanticAnalyzerAgent": "agent instance with a ReAct executor bound to search tools"
    }
    """

    ## Initialize the agent with a ReAct executor using semantic search tools
    def __init__(self, llm):
        """
        args   : {
            "llm": "language model compatible with create_react_agent"
        }
        return : {
            "None": "initializes self.agent"
        }
        """
        self.agent = create_react_agent(model=llm, tools=[dense_search, sparse_search, hybrid_search])