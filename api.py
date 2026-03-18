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
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.v1.router import api_router
from api import sse
from graph.main_graph import create_main_graph
from schema import ChatRequest, Message, NODE_TO_AGENT

# Ensure the MultiAgentRAG package root is importable
sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="MultiAgentRAG API", version="1.0.0")
app.include_router(router=api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
