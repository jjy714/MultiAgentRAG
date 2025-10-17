from langchain_core.tools import tool
from langgraph.types import Command

@tool
def transfer_tool(dest_agent: str, passing_data):
    return Command(
        goto=dest_agent,
        update={"my_state_key": "my_state_value"},
        graph=Command.PARENT,
    )