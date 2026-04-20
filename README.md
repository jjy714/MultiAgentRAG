# DMA-RAG

A multi-agent RAG system built with FastAPI and LangGraph. Queries are classified by a discussion panel into three complexity tiers (easy / medium / complex), then routed to tier-specific subgraphs for retrieval-augmented answering. Responses are streamed to the frontend over SSE.

## Architecture

### Discussion Panel
Three advocate agents run in parallel — **DirectResponderAdvocate**, **ContextualAnalystAdvocate**, and **DeepResearcherAdvocate** — each arguing for a different tier. A **ModeratorAgent** tallies the votes and decides the routing tier.

### Easy Tier
**DirectResponderAgent** answers from LLM knowledge. If it sets `web_needed=true`, **WebSearchAgent** runs a live Exa search and merges the result.

### Medium Tier
**RetrieverAgent** (Qdrant vector search) and **WebSearchAgent** run in parallel. **ExtractorAgent** processes the retrieved documents, then **QuestionAnsweringAgent** synthesises the final answer.

### Complex Tier
**PlannerAgent** decomposes the query into up to five sub-questions. **StepDefinerAgent** loops through each step, dispatching to either a RAG subgraph or a web search subgraph. After all steps complete, **StepDefinerAgent** calls **PlanSummarizerAgent** to produce the final answer.

```
User query
  └─ DiscussionPanel (3 advocates + moderator)
       ├─ easy    → DirectResponderAgent [→ WebSearchAgent]
       ├─ medium  → RetrieverAgent ∥ WebSearchAgent → ExtractorAgent → QAAgent
       └─ complex → PlannerAgent → StepDefiner loop → PlanSummarizer
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
cd DMA-RAG
uv sync
```

Create `.env.dev` with your credentials:

```
LLM_BACKEND=gemini
GEMINI_MODEL_NAME=gemini-2.5-flash-lite
GOOGLE_API_KEY=<your_key>

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=kilt_dpr_100w
QDRANT_PORT=6333

SERVER_HOST=localhost
EMBEDDING_PORT=8080

# Optional — for web search
EXA_SEARCH_API_KEY=<your_key>
```

### Qdrant

Start Qdrant locally (Docker recommended):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

To populate a test collection with example documents:

```bash
uv run python tools/setup_qdrant.py
```

### Embedding server

The retriever calls a vllm-compatible embedding endpoint. Start it with:

```bash
bash model_serve.sh
```

Or set `EMBEDDING_PORT` to point at an existing server.

## Running

```bash
uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Accepts `{"message": "...", "history": [...]}`, returns SSE stream |
| `GET` | `/api/health` | Returns `{"status": "ok"}` |

### SSE event types

| Type | Fields | Description |
|---|---|---|
| `agent_start` | `agent` | An agent node began executing |
| `token` | `agent`, `content` | Streamed text chunk |
| `agent_end` | `agent` | An agent node finished |
| `vote` | `agent`, `vote`, `reasoning` | Advocate vote from the discussion panel |
| `routing` | `complexity` | Routing decision (`easy` / `medium` / `complex`) |
| `done` | `final_answer` | Stream complete |
| `error` | `message` | Backend error |

## Project Structure

```
DMA-RAG/
├── app.py                      # FastAPI entry point
├── main.py                     # CLI runner (dev/debug)
├── agents/
│   ├── Prompts/                # YAML prompt files for each agent
│   ├── llm.py                  # LLM factory (Gemini or local vllm)
│   ├── token_utils.py          # Token usage helpers
│   ├── load_prompt.py          # YAML prompt loader
│   ├── retriever_node.py       # Qdrant retriever + embedding cache
│   ├── *Advocate.py            # Discussion panel advocate agents
│   ├── ModeratorAgent.py       # Vote tallier
│   ├── DirectResponderAgent.py # Easy tier answerer
│   ├── WebSearchAgent.py       # Exa web search agent
│   ├── ExtractorAgent.py       # Document extraction agent
│   ├── QuestionAnsweringAgent.py
│   ├── PlannerAgent.py
│   └── StepDefinerAgent.py
├── graph/
│   ├── main_graph.py           # Top-level graph with discussion + tier routing
│   ├── discussion_graph.py     # Fan-out advocate graph
│   ├── easy_graph.py
│   ├── medium_graph.py
│   ├── complex_graph.py
│   ├── run_graph.py            # SSE event emitter
│   └── state/                  # TypedDict state definitions
├── api/                        # FastAPI routers and SSE helper
├── schema/                     # Pydantic request models + NODE_TO_AGENT map
└── tools/                      # Qdrant setup, upload scripts
```
