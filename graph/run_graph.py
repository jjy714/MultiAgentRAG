from typing import AsyncGenerator
from api import sse
from graph.main_graph import create_main_graph
from schema import ChatRequest, NODE_TO_AGENT


## Invoke the real MultiAgentRAG main graph and stream SSE events per node lifecycle

async def run_graph(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    args   : {
        "request (ChatRequest)": "incoming chat request containing the user message"
    }
    return : {
        "AsyncGenerator[str, None]": "async generator yielding SSE-formatted strings"
    }
    """
    
    graph = create_main_graph()

    initial_state = {
        "original_question": request.message,
        "complexity": "",
        "final_answer": None,
        "documents": [],
        "doc_ids": [],
        "notes": [],
        "plan": [],
        "step_question": [],
        "step_output": [],
        "step_notes": [],
        "is_report": False,
        "plan_summary": None,
        "web_needed": False,
        "web_result": None,
    }

    active_nodes: set[str] = set()
    final_answer = ""

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            node_name = event.get("name", "")
            agent_id = NODE_TO_AGENT.get(node_name, node_name)

            # Node starts
            if kind == "on_chain_start" and node_name in NODE_TO_AGENT:
                if node_name not in active_nodes:
                    active_nodes.add(node_name)
                    yield sse({"type": "agent_start", "agent": agent_id})

            # State updates — emit routing decision when complexity is set
            elif kind == "on_chain_stream" and node_name == "discussion_panel":
                chunk = event.get("data", {}).get("chunk", {})
                complexity = chunk.get("complexity", "")
                if complexity:
                    yield sse({"type": "routing", "complexity": complexity})

            # LLM token streaming
            elif kind == "on_chat_model_stream":
                content = (
                    event.get("data", {})
                         .get("chunk", {})
                         .get("content", "")
                )
                if content:
                    # Find the closest parent node for attribution
                    tags = event.get("tags", [])
                    parent_agent = "AssistantAgent"
                    for tag in tags:
                        if tag in NODE_TO_AGENT:
                            parent_agent = NODE_TO_AGENT[tag]
                            break
                    yield sse({"type": "token", "agent": parent_agent, "content": content})

            # Node ends
            elif kind == "on_chain_end" and node_name in NODE_TO_AGENT:
                if node_name in active_nodes:
                    active_nodes.discard(node_name)
                    yield sse({"type": "agent_end", "agent": agent_id})

                # Capture final answer from terminal tier nodes
                if node_name in ("easy_tier", "medium_tier", "complex_tier"):
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        ans = output.get("final_answer", "")
                        if ans:
                            final_answer = ans

        yield sse({"type": "done", "final_answer": final_answer})

    except Exception as exc:
        yield sse({"type": "error", "message": str(exc)})
        yield sse({"type": "done", "final_answer": ""})
