"""
MultiAgentRAG FastAPI Backend
==============================
Streams SSE events from the real LangGraph graph to the React frontend.

SSE event format (newline-delimited JSON after "data: " prefix):
  { "type": "agent_start", "agent": "<AgentName>", "tier": "<easy|medium|complex>" }
  { "type": "token",       "agent": "<AgentName>", "content": "..." }
  { "type": "agent_end",   "agent": "<AgentName>" }
  { "type": "routing",     "complexity": "<easy|medium|complex>" }
  { "type": "done",        "final_answer": "..." }
  { "type": "error",       "message": "..." }

Agents emitted (match frontend AgentStatusPanel):
  Discussion panel:  DiscussionPanel
  Easy tier:         DirectResponderAgent, WebSearchAgent
  Medium tier:       RetrieverAgent, ExtractorAgent, QuestionAnsweringAgent
  Complex tier:      PlannerAgent, StepDefinerAgent, RetrieverAgent,
                     ExtractorAgent, QuestionAnsweringAgent, WebSearchAgent
"""

import asyncio
import json
import sys
import os
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from schema import ChatRequest, Message

# Ensure the MultiAgentRAG package root is importable
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(title="MultiAgentRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Agent name → display label mapping (for SSE events)
# ---------------------------------------------------------------------------

# Maps LangGraph node names to human-readable agent IDs used by the frontend
NODE_TO_AGENT = {
    # Discussion panel
    "discussion_panel":             "DiscussionPanel",
    "direct_responder_advocate":    "DirectResponderAdvocate",
    "contextual_analyst_advocate":  "ContextualAnalystAdvocate",
    "deep_researcher_advocate":     "DeepResearcherAdvocate",
    "moderator":                    "ModeratorAgent",
    # Easy tier
    "direct_responder":             "DirectResponderAgent",
    "web_search":                   "WebSearchAgent",
    "easy_tier":                    "EasyTier",
    # Medium tier
    "retrieve":                     "RetrieverAgent",
    "extract":                      "ExtractorAgent",
    "answer":                       "QuestionAnsweringAgent",
    "medium_tier":                  "MediumTier",
    # Complex tier
    "planner":                      "PlannerAgent",
    "executor":                     "StepDefinerAgent",
    "complex_tier":                 "ComplexTier",
    "rag_execute":                  "RetrieverAgent",
    "web_execute":                  "WebSearchAgent",
}


# ---------------------------------------------------------------------------
# Real graph runner
# ---------------------------------------------------------------------------

async def run_graph(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Invokes the real MultiAgentRAG main graph and streams SSE events.
    Uses LangGraph's astream_events to emit per-node lifecycle events.
    """
    from graph.main_graph import create_main_graph

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

            # ── Node starts ──────────────────────────────────────────────
            if kind == "on_chain_start" and node_name in NODE_TO_AGENT:
                if node_name not in active_nodes:
                    active_nodes.add(node_name)
                    yield sse({"type": "agent_start", "agent": agent_id})

            # ── State updates — emit routing decision when complexity is set ─
            elif kind == "on_chain_stream" and node_name == "discussion_panel":
                chunk = event.get("data", {}).get("chunk", {})
                complexity = chunk.get("complexity", "")
                if complexity:
                    yield sse({"type": "routing", "complexity": complexity})

            # ── LLM token streaming ──────────────────────────────────────
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

            # ── Node ends ────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "MultiAgentRAG API"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_graph(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
