# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Multi-Agent RAG (Retrieval-Augmented Generation) system with complexity-based routing. Queries are classified into three tiers (easy/medium/complex) by a discussion panel of advocate agents, then routed to tier-specific subgraphs for processing.

- **DMA-RAG/** — Main backend (FastAPI + LangGraph). This is the primary implementation.
- **MA-RAG/** — Earlier prototype (FastAPI + LangGraph with Google GenAI). Reference only.
- **frontend/** — React 18 + Vite UI for real-time agent monitoring.
- **Evaluation/** — Benchmarking harness comparing both systems.

## Commands

### Backend (DMA-RAG)
```bash
cd DMA-RAG
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install       # first time
npm run dev       # http://localhost:5173
npm run build     # production build
```

### Evaluation
```bash
cd Evaluation
uv run python evaluate.py                                  # run all questions against both systems
uv run python evaluate.py --system multi_agent_rag         # single system
uv run python evaluate.py --max-questions 10               # limit questions
uv run python evaluate.py --no-bertscore                   # skip slow BERTScore metric
```

### Full stack startup
```bash
bash run_capstone.sh   # or start each service manually in separate terminals
```

## Architecture

### Three-Tier Routing (DMA-RAG)

All queries first pass through the **Discussion Panel** (`discussion_graph.py`): three advocate agents (DirectResponder, ContextualAnalyst, DeepResearcher) debate in parallel, then a Moderator agent decides complexity. This emits a `routing` SSE event with the chosen tier.

**Easy** — `easy_graph.py`: DirectResponderAgent answers from LLM knowledge; optionally WebSearchAgent for current info.

**Medium** — `medium_graph.py`: RetrieverAgent → ExtractorAgent → QuestionAnsweringAgent (with parallel WebSearchAgent).

**Complex** — `complex_graph.py`: PlannerAgent → StepDefinerAgent → per-step retrieval+extraction+QA loop → PlanSummarizerAgent synthesizes all steps.

### State & Graph Structure
- `GraphState` (in `graph_state.py`) is the shared state type, passed through all subgraphs
- `sum_dicts()` custom reducer accumulates token usage across parallel agent executions
- Subgraphs are composed into `main_graph.py` via LangGraph's `add_node` with compiled subgraphs
- `run_graph.py` wraps graph execution and emits SSE events (`agent_start`, `token`, `agent_end`, `routing`, `done`, `error`)

### Prompt Management
All agent prompts live in `DMA-RAG/agents/Prompts/` as YAML files. Loaded at runtime via `load_prompt.py`. To change agent behavior, edit the corresponding `.yaml` file rather than Python code.

### API
- `POST /api/chat` — accepts `{"query": "..."}`, responds with SSE stream
- `GET /api/health` — health check
- CORS configured for `localhost:5173` and `localhost:3000`

### Frontend SSE Integration
The `useAgentStream()` hook in `frontend/src/hooks/` manages the SSE connection and parses event types into state. The `AgentStatusPanel` component displays live agent activity by tier. Vite proxies `/api/*` to `http://localhost:8000`.

### Environment
Backend requires a `.env.dev` file in `DMA-RAG/` with OpenAI API key and Qdrant connection settings. Qdrant is the vector database used for all semantic retrieval.

### Evaluation Metrics
Outputs per-question and averaged: `latency_s`, `total_tokens`, `rouge1_f1`, `rouge2_f1`, `rougeL_f1`, `bert_f1`, and `complexity_tier` (DMA-RAG only). Results written as timestamped JSON to `Evaluation/results/`.
implement it all. when you're done with a task or phase, mark it as completed in the
plan document. do not stop until all tasks and phases are completed. do not add
unnecessary comments or jsdocs, do not use any or unknown types. continuously run
typecheck to make sure you're not introducing new issues.