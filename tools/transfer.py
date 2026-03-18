from langchain_core.tools import tool
from langgraph.types import Command

## LangChain tool that transfers graph execution to a destination agent with optional state update
@tool
def transfer_tool(dest_agent: str, passing_data):
    """
    args   : {
        "dest_agent (str)": "name of the target agent node to transfer control to",
        "passing_data": "data to pass along with the transfer command"
    }
    return : {
        "Command": "LangGraph Command object that redirects execution to the destination agent"
    }
    """
    return Command(
        goto=dest_agent,
        update={"my_state_key": "my_state_value"},
        graph=Command.PARENT,
    )