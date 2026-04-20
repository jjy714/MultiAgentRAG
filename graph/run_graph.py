from typing import AsyncGenerator
from api import sse
from graph.main_graph import create_main_graph
from schema import ChatRequest, NODE_TO_AGENT


async def run_graph(request: ChatRequest) -> AsyncGenerator[str, None]:
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
        "token_usage": {},
    }

    active_nodes: set[str] = set()
    in_discussion = False
    final_answer = ""

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            node_name = event.get("name", "")
            agent_id = NODE_TO_AGENT.get(node_name, node_name)

            if kind == "on_chain_start" and node_name in NODE_TO_AGENT:
                if node_name not in active_nodes:
                    active_nodes.add(node_name)
                    yield sse({"type": "agent_start", "agent": agent_id})
                if node_name == "discussion_panel":
                    in_discussion = True

            elif kind == "on_chain_stream" and node_name == "discussion_panel":
                chunk = event.get("data", {}).get("chunk", {})
                complexity = (
                    chunk.get("complexity", "")
                    if isinstance(chunk, dict)
                    else getattr(chunk, "complexity", "")
                )
                if complexity:
                    yield sse({"type": "routing", "complexity": complexity})

            # Stream LLM tokens to the client only during execution (not discussion).
            # The discussion panel votes are surfaced separately via vote events.
            elif kind == "on_chat_model_stream" and not in_discussion:
                data = event.get("data", {})
                chunk = data.get("chunk")

                if chunk:
                    content = (
                        chunk.content
                        if hasattr(chunk, "content")
                        else chunk.get("content", "")
                    )

                    if content:
                        tags = event.get("tags", [])
                        parent_agent = "AssistantAgent"
                        for tag in tags:
                            if tag in NODE_TO_AGENT:
                                parent_agent = NODE_TO_AGENT[tag]
                                break
                        yield sse(
                            {"type": "token", "agent": parent_agent, "content": content}
                        )

            elif kind == "on_chain_end" and node_name in NODE_TO_AGENT:
                if node_name in active_nodes:
                    active_nodes.discard(node_name)
                    yield sse({"type": "agent_end", "agent": agent_id})
                if node_name == "discussion_panel":
                    in_discussion = False

                if node_name in (
                    "direct_responder_advocate",
                    "contextual_analyst_advocate",
                    "deep_researcher_advocate",
                ):
                    output = event.get("data", {}).get("output", {})
                    votes = output.get("votes", []) if isinstance(output, dict) else []
                    if votes:
                        v = votes[0]
                        yield sse(
                            {
                                "type": "vote",
                                "agent": agent_id,
                                "vote": v.get("vote", "medium"),
                                "reasoning": v.get("reasoning", ""),
                            }
                        )

                if node_name in ("easy_tier", "medium_tier", "complex_tier"):
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        final_answer = output.get("final_answer", "")

        yield sse({"type": "done", "final_answer": final_answer})

    except Exception as exc:
        yield sse({"type": "error", "message": str(exc)})
        yield sse({"type": "done", "final_answer": ""})
