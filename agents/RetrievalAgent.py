from Agent import Agent
from tools import dense_search, sparse_search, hybrid_search
from langchain.agents import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


### Agent class that performs document retrieval using dense, sparse, and hybrid search strategies
class RetrievalAgent(Agent):
    """
    args   : {
        "llm": "the underlying language model used by the ReAct agent"
    }
    return : {
        "RetrievalAgent": "agent instance with a ReAct executor bound to retrieval tools"
    }
    """

    ## Initialize the agent with a ReAct executor using dense, sparse, and hybrid search tools
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


# Path to the extractor tool server (MCP stdio server)
EXTRACTOR_TOOL_PATH = "tools"

## Async function that connects to an MCP stdio server and invokes an extraction agent
async def ExtractorAgent(state):
    """
    args   : {
        "state (dict)": "agent state containing input for the MCP-connected extraction agent"
    }
    return : {
        "dict": "state updated with agent_response from the MCP tool agent"
    }
    """
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        # Make sure to update to the full absolute path to your math_server.py file
        args=["/path/to/math_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_agent("openai:gpt-4.1", tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

    return {state["agent_response"]: agent_response}