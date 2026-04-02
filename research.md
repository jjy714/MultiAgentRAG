# MultiAgentRAG — Comprehensive System Research

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture: The Big Picture](#3-architecture-the-big-picture)
4. [State Management](#4-state-management)
5. [The Discussion Panel — Complexity Routing](#5-the-discussion-panel--complexity-routing)
6. [The Main Graph — Orchestration Layer](#6-the-main-graph--orchestration-layer)
7. [Easy Tier — Direct Response](#7-easy-tier--direct-response)
8. [Medium Tier — Single-Pass RAG](#8-medium-tier--single-pass-rag)
9. [Complex Tier — Multi-Step Plan Executor](#9-complex-tier--multi-step-plan-executor)
10. [Shared Agents and Utilities](#10-shared-agents-and-utilities)
11. [Prompt Engineering System](#11-prompt-engineering-system)
12. [External Integrations](#12-external-integrations)
13. [API Layer and SSE Streaming](#13-api-layer-and-sse-streaming)
14. [Token Usage Tracking](#14-token-usage-tracking)
15. [Data Flow Walkthroughs](#15-data-flow-walkthroughs)
16. [Design Patterns and Architectural Decisions](#16-design-patterns-and-architectural-decisions)
17. [Configuration and Environment](#17-configuration-and-environment)
18. [Dependency Stack](#18-dependency-stack)

---

## 1. Overview

**MultiAgentRAG** is a LangGraph-based multi-agent system that routes user queries to one of three processing tiers — Easy, Medium, or Complex — based on a three-advocate voting panel. Each tier applies a different retrieval-augmented generation (RAG) strategy proportional to the query's complexity:

- **Easy**: Direct LLM knowledge, optionally supplemented by a live web search.
- **Medium**: Single-pass RAG combining vector database retrieval with a web search, processed in parallel.
- **Complex**: Full multi-step planning — the query is decomposed into sub-questions, each executed in sequence with its own RAG or web search pass, then synthesized into a final answer.

The system exposes a FastAPI backend that streams real-time SSE events (agent start/end, tokens, routing decisions, final answer) to a frontend such as a React application.

---

## 2. Repository Structure

```
MultiAgentRAG/
├── app.py                        # FastAPI app entry point
├── main.py                       # CLI entry point
├── .env.dev                      # Environment variable configuration
├── pyproject.toml                # Python dependencies
│
├── agents/
│   ├── llm.py                    # Shared LLM factory (get_llm)
│   ├── load_prompt.py            # YAML prompt loader utilities
│   ├── retriever_node.py         # Qdrant vector retrieval node
│   ├── DirectResponderAdvocate.py
│   ├── ContextualAnalystAdvocate.py
│   ├── DeepResearcherAdvocate.py
│   ├── ModeratorAgent.py
│   ├── DirectResponderAgent.py
│   ├── WebSearchAgent.py
│   ├── ExtractorAgent.py
│   ├── QuestionAnsweringAgent.py
│   ├── PlannerAgent.py
│   ├── StepDefinerAgent.py
│   └── Prompts/
│       ├── DirectResponderAdvocate.yaml
│       ├── ContextualAnalystAdvocate.yaml
│       ├── DeepResearcherAdvocate.yaml
│       ├── DirectResponderAgent.yaml
│       ├── ExtractorAgent.yaml
│       ├── QuestionAnsweringAgent.yaml
│       ├── PlannerAgent.yaml
│       ├── StepDefinerAgent.yaml
│       └── PlanSummarizerAgent.yaml
│
├── graph/
│   ├── main_graph.py             # Top-level orchestration graph
│   ├── discussion_graph.py       # 3-advocate panel subgraph
│   ├── easy_graph.py             # Easy tier subgraph
│   ├── medium_graph.py           # Medium tier subgraph
│   ├── complex_graph.py          # Complex tier + plan executor subgraphs
│   ├── run_graph.py              # SSE streaming graph runner
│   └── state/
│       ├── GraphState.py         # Main shared state (sum_dicts reducer)
│       ├── DiscussionState.py    # Discussion panel state
│       ├── RagState.py           # Single-pass RAG state
│       ├── PlanExecState.py      # Plan executor loop state
│       ├── PlanState.py          # Planner output schema
│       ├── QAanswerState.py      # QA agent output schema
│       └── StepTaskState.py      # Step definer output schema
│
├── api/
│   ├── sse.py                    # SSE serializer
│   └── v1/
│       ├── router.py             # API router
│       └── endpoints/
│           ├── chat.py           # POST /api/chat
│           └── health.py         # GET /api/health
│
└── schema/
    ├── __init__.py               # NODE_TO_AGENT mapping + re-exports
    ├── ChatRequest.py            # Pydantic request model
    └── Message.py                # Pydantic message model
```

---

## 3. Architecture: The Big Picture

The system is composed of a hierarchy of LangGraph `StateGraph` instances. Each graph is compiled independently and invoked as a node inside a parent graph. This nesting is what gives the system its modularity.

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN GRAPH                              │
│                                                                 │
│   START ──► [discussion_panel] ──► route_by_complexity          │
│                                         │                       │
│                           ┌────────────┼────────────┐          │
│                           ▼            ▼            ▼          │
│                      [easy_tier]  [medium_tier] [complex_tier]  │
│                           │            │            │           │
│                           └────────────┼────────────┘          │
│                                        ▼                       │
│                                       END                       │
└─────────────────────────────────────────────────────────────────┘
```

Each tier node is itself a compiled subgraph:

```
discussion_panel  →  create_discussion_graph()
easy_tier         →  create_easy_graph()
medium_tier       →  create_medium_graph()
complex_tier      →  create_complex_graph()
                        └─► complex_executor_node
                                └─► create_plan_executor_graph()
                                        └─► create_single_rag_graph()
                                        └─► create_single_web_graph()
```

All graphs use the `asyncio` compatible `ainvoke` / `astream_events` methods, enabling non-blocking concurrent execution and streaming.

---

## 4. State Management

State is the backbone of the system. LangGraph passes state through all nodes; nodes return partial dicts that are merged into the running state by reducers.

### 4.1 Custom Reducer: `sum_dicts`

```python
# graph/state/GraphState.py
def sum_dicts(d1: dict, d2: dict) -> dict:
    if not d1: return d2
    if not d2: return d1
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}
```

This reducer is attached to every `token_usage` field via `Annotated[dict, sum_dicts]`. It allows multiple parallel nodes to each return their own token usage dicts, which are then automatically summed together by the LangGraph state merge.

### 4.2 `GraphState` — The Primary Shared State

`GraphState` is used by the main graph and all tier wrapper nodes. It carries everything needed across all tiers.

| Field | Type | Tier | Description |
|---|---|---|---|
| `original_question` | `str` | All | The user's raw query |
| `complexity` | `str` | All | Routing decision: `"easy"`, `"medium"`, `"complex"` |
| `final_answer` | `Optional[str]` | All | The output produced by whichever tier ran |
| `token_usage` | `Annotated[dict, sum_dicts]` | All | Accumulated token counts, summed by reducer |
| `documents` | `List[str]` | Medium, Complex | Retrieved document page contents |
| `doc_ids` | `List[str]` | Medium, Complex | Qdrant chunk IDs for retrieved documents |
| `notes` | `List[str]` | Medium, Complex | Extractor output (bullet summaries) |
| `plan` | `List[str]` | Complex | Ordered list of sub-questions from PlannerAgent |
| `step_question` | `List[dict]` | Complex | StepTaskState dicts per plan step |
| `step_output` | `Annotated[List[StepOutput], operator.add]` | Complex | Accumulated per-step results |
| `step_notes` | `Annotated[List[str], operator.add]` | Complex | Accumulated notes from all steps |
| `is_report` | `bool` | Complex | Whether to format output as a formal report |
| `plan_summary` | `Optional[str]` | Complex | Final synthesized answer from PlanSummarizerAgent |
| `web_needed` | `bool` | Easy | Whether DirectResponderAgent flagged need for web search |
| `web_result` | `Optional[str]` | Easy | Raw result from web search |

The `operator.add` reducer on `step_output` and `step_notes` means these lists are concatenated (not overwritten) each time the loop appends a new step result.

### 4.3 `DiscussionState`

Used only inside the discussion panel subgraph.

```python
class DiscussionState(TypedDict):
    question: str
    votes: Annotated[List[AdvocateVote], operator.add]  # fan-in reducer
    complexity: str
    token_usage: Annotated[dict, sum_dicts]
```

The `operator.add` reducer on `votes` is the key to the fan-out/fan-in pattern: three parallel advocate nodes each return `{"votes": [their_vote]}`, and these three single-element lists are automatically concatenated into one 3-element list by the time `ModeratorAgent` reads from state.

### 4.4 `RagState`

Used by the medium tier graph and by both single-step executor subgraphs inside the complex tier.

```python
class RagState(TypedDict):
    question: str
    documents: List[str]
    doc_ids: List[str]
    notes: List[str]
    final_raw_answer: Optional[dict]   # QAAnswerState
    token_usage: Annotated[dict, sum_dicts]
```

### 4.5 `PlanExecState`

Used by the plan-executor loop graph in the complex tier.

```python
class PlanExecState(TypedDict):
    original_question: str
    plan: List[str]
    step_question: Annotated[List[dict], operator.add]  # wait: actually List[dict] plain
    step_output: Annotated[List[dict], operator.add]    # accumulates per step
    step_notes: Annotated[List[str], operator.add]      # accumulates per step
    stop: bool
    plan_summary: Optional[str]
    token_usage: Annotated[dict, sum_dicts]
```

Note: `step_output` and `step_notes` use `operator.add` — each iteration of the loop appends rather than replaces.

### 4.6 Supporting TypedDicts

| Schema | Fields | Purpose |
|---|---|---|
| `AdvocateVote` | `role`, `vote`, `reasoning` | Single advocate ballot |
| `QAAnswerState` | `answer`, `confidence`, `sources` | QA agent output |
| `StepOutput` | `task`, `answer`, `notes` | One completed plan step |
| `PlanState` | `step: List[str]`, `is_report: bool` | Planner output |
| `StepTaskState` | `task: str`, `type: str` | StepDefiner output |

---

## 5. The Discussion Panel — Complexity Routing

### 5.1 Graph Topology

```python
# graph/discussion_graph.py
graph.add_edge(START, "direct_responder_advocate")    # ─┐
graph.add_edge(START, "contextual_analyst_advocate")  #  ├─ parallel fan-out
graph.add_edge(START, "deep_researcher_advocate")     # ─┘

graph.add_edge("direct_responder_advocate",   "moderator")  # ─┐
graph.add_edge("contextual_analyst_advocate", "moderator")  #  ├─ fan-in
graph.add_edge("deep_researcher_advocate",    "moderator")  # ─┘

graph.add_edge("moderator", END)
```

Three advocate nodes run in parallel (LangGraph executes nodes with no incoming dependency simultaneously). All three converge at `moderator` once finished.

### 5.2 Advocate Agents

Each advocate is an async function that calls the shared `get_llm()` and returns a single `AdvocateVote`. They are structurally identical but use different prompts with different voting biases.

**Staggered delays** (to avoid Google API rate-limit 429 errors):

| Advocate | Sleep range |
|---|---|
| `DirectResponderAdvocate` | 0.1 – 1.0 s |
| `ContextualAnalystAdvocate` | 0.5 – 2.0 s |
| `DeepResearcherAdvocate` | 1.0 – 3.0 s |

Each advocate:
1. Reads `state["question"]`
2. Loads its YAML prompt
3. Invokes the LLM (async)
4. Strips markdown code fences from the response if present (`````json ... `````)
5. Parses the JSON into `AdvocateVote`
6. Falls back to a hardcoded vote if JSON parsing fails
7. Returns `{"votes": [result], "token_usage": ...}`

**Voting Criteria:**

| Advocate | Primary Vote | Conditions |
|---|---|---|
| `DirectResponderAdvocate` | `"easy"` | General knowledge, quick web search suffices, no document retrieval needed |
| `ContextualAnalystAdvocate` | `"medium"` | Requires internal knowledge base, single retrieval pass sufficient |
| `DeepResearcherAdvocate` | `"complex"` | Multi-step, dependent sub-questions, comparison/synthesis required |

Each advocate can vote for any tier — the bias just comes from their prompt framing their preferred tier as the default case.

### 5.3 Moderator Agent

```python
# agents/ModeratorAgent.py
def ModeratorAgent(state: DiscussionState) -> dict:
    votes = state.get("votes", [])
    vote_counts = Counter(v.get("vote", "medium") for v in votes)
    majority = vote_counts.most_common(1)[0]
    winning_tier, winning_count = majority
    if winning_count == 1:          # 3-way tie (all different)
        winning_tier = "medium"
    return {"complexity": winning_tier}
```

The moderator does **no LLM call**. It is a pure Python function. It tallies votes with `Counter`, takes the most common, and falls back to `"medium"` when all three votes differ (each gets count=1).

Possible vote distributions:
- `3-0-0` → clear winner
- `2-1-0` → clear winner
- `1-1-1` → tie → defaults to `"medium"`

---

## 6. The Main Graph — Orchestration Layer

```python
# graph/main_graph.py
graph.add_edge(START, "discussion_panel")
graph.add_conditional_edges(
    "discussion_panel",
    route_by_complexity,          # reads state["complexity"]
    {"easy": "easy_tier", "medium": "medium_tier", "complex": "complex_tier"},
)
graph.add_edge("easy_tier",    END)
graph.add_edge("medium_tier",  END)
graph.add_edge("complex_tier", END)
```

The `discussion_panel` node wraps `create_discussion_graph()`, invoking it as a nested subgraph:

```python
async def discussion_panel_node(state: GraphState) -> dict:
    discussion_graph = create_discussion_graph()
    result = await discussion_graph.ainvoke({
        "question": state["original_question"],
        "votes": [], "complexity": "", "token_usage": {}
    })
    return {"complexity": result["complexity"], "token_usage": result["token_usage"]}
```

The three tier nodes (`easy_node`, `medium_node`, `complex_node`) each create their subgraph fresh, invoke it, and return `{"final_answer": ..., "token_usage": ...}`.

**Note**: `medium_node` builds a separate `RagState` input dict because the medium tier uses `RagState` internally (not `GraphState`), remapping `original_question` → `question`.

---

## 7. Easy Tier — Direct Response

### 7.1 Graph Topology

```
START ──► [direct_responder] ──► _route_web ──► "web_search" ──► END
                                              └─► END (if no web needed)
```

### 7.2 DirectResponderAgent

**File**: `agents/DirectResponderAgent.py`
**Type**: Synchronous LangGraph node
**Input state field**: `original_question`

This agent answers from LLM training knowledge. Its prompt instructs the model to:
- Answer immediately if possible
- Set `web_needed: true` only for: current events, live prices, specific URLs, real-time statistics

**Output** (JSON parsed from LLM response):
```python
class DirectResponderOutput(TypedDict):
    answer: str
    web_needed: bool
```

Returns: `{"final_answer": answer, "web_needed": web_needed, "token_usage": ...}`

### 7.3 Web Search Fallback

If `state["web_needed"] == True`, the graph routes to `_web_search_then_finalize`:

```python
async def _web_search_then_finalize(state: GraphState) -> dict:
    web_result = await WebSearchAgent({"question": original_question})
    web_content = web_result["documents"][0]
    merged = f"{state['final_answer']}\n\n[Web Context]\n{web_content}".strip()
    return {"final_answer": merged, "web_result": web_content}
```

The initial LLM answer is kept and the web context is appended as a `[Web Context]` section.

---

## 8. Medium Tier — Single-Pass RAG

### 8.1 Graph Topology

```
START ──► [retrieve]  ──► [extract] ──► [answer] ──► END
      └─► [web_search] ──────────────────►
```

Retrieval and web search run in **parallel** from START. Both feed into the `answer` node. LangGraph waits for all incoming edges to be satisfied before executing a node, so `answer` waits for both `extract` and `web_search` to complete.

### 8.2 retriever_node

**File**: `agents/retriever_node.py`

Wraps a `Retriever` class that:
1. Connects to Qdrant at `localhost:6333`
2. Calls an embedding HTTP service at `http://{SERVER_HOST}:{EMBEDDING_PORT}/v1/embed` with `{"query": query}`
3. Receives a 1024-dimensional embedding vector
4. Calls `qdrant_client.query_points()` with cosine similarity, returning top-5 results

The payload of each Qdrant point is expected to have:
- `chunk_id`: string identifier
- `page_content`: the actual text

**Returns**: `{"question": query, "documents": [page_content, ...], "doc_ids": [chunk_id, ...]}`

The node also handles a special case: if `state["question"]` is a dict (when called from complex tier), it extracts `.get("task", "")` as the query string.

### 8.3 ExtractorAgent

**File**: `agents/ExtractorAgent.py`
**Type**: Synchronous node

Receives `state["documents"]` and `state["question"]`. Passes them to the LLM with a prompt instructing it to extract relevant facts as concise bullet points or a short paragraph. Appends the result to `state["notes"]`.

Returns: `{"notes": updated_notes_list, "token_usage": ...}`

### 8.4 WebSearchAgent

**File**: `agents/WebSearchAgent.py`
**Type**: Async node

Uses `langchain_mcp_adapters.client.MultiServerMCPClient` to connect to the Exa search MCP server via `streamable_http` transport. Creates a **ReAct agent** (`create_react_agent`) that receives the Exa tools and the user question.

The ReAct agent autonomously decides how to call the Exa search tool(s), reasons about the results, and produces a final answer. The agent response messages are processed to:
1. Extract the last `AIMessage` content as the search result
2. Strip `<think>...</think>` tags (chain-of-thought from reasoning models)
3. Aggregate token usage from all AIMessage metadata

Returns: `{"documents": [search_result_text], "token_usage": ...}`

### 8.5 QuestionAnsweringAgent

**File**: `agents/QuestionAnsweringAgent.py`
**Type**: Synchronous node

Synthesizes the final answer from context. The context is built as:
- If `notes` are present: `"\n\n".join(notes)` (preferred: uses extracted summaries)
- Otherwise: `"\n\n".join(documents)` (raw retrieved docs)

The LLM is prompted to rate its confidence:
- `"high"` — context directly answers the question
- `"medium"` — context partially addresses the question
- `"low"` — context is insufficient

**Output** (`QAAnswerState`):
```python
{"answer": str, "confidence": "high"|"medium"|"low", "sources": List[str]}
```

Returns: `{"final_raw_answer": QAAnswerState, "token_usage": ...}`

The medium tier's wrapper node then extracts `final_raw_answer["answer"]` as `final_answer`.

---

## 9. Complex Tier — Multi-Step Plan Executor

The complex tier is the most elaborate part of the system. It is implemented as three nested layers of subgraphs.

### 9.1 Top-Level Complex Graph

```python
# graph/complex_graph.py — create_complex_graph()
g.add_edge(START, "planner")
g.add_edge("planner", "executor")
g.add_edge("executor", END)
```

`planner` → `PlannerAgent` (decomposes query)
`executor` → `complex_executor_node` (runs the plan)

### 9.2 PlannerAgent

**File**: `agents/PlannerAgent.py`
**Type**: Synchronous node
**Input**: `state["original_question"]`

The planner decomposes the question into an **ordered list of sub-questions** using the following rules (from its prompt):
- Non-overlapping, non-redundant sub-questions
- Ordered so later sub-questions can use earlier answers
- Each sub-question is independently solvable by either retrieval or web search
- No verification or meta-questions ("is this correct?")
- `is_report: true` only if the user explicitly requests a report/document

**Output** (`PlanState`):
```python
{"step": ["sub-question 1", "sub-question 2", ...], "is_report": bool}
```

Returns: `{"plan": List[str], "is_report": bool, "token_usage": ...}`

### 9.3 complex_executor_node

This is an async node that bridges the top-level `GraphState` into the plan executor subgraph:

```python
async def complex_executor_node(state: GraphState) -> dict:
    plan_executor = create_plan_executor_graph()
    result = await plan_executor.ainvoke({
        "original_question": state["original_question"],
        "plan": state["plan"],
        "step_question": [], "step_output": [],
        "step_notes": [], "stop": False, "plan_summary": None, "token_usage": {}
    })
    return {"final_answer": result["plan_summary"], "token_usage": result["token_usage"]}
```

### 9.4 Plan Executor Loop Graph

```python
# graph/complex_graph.py — create_plan_executor_graph()
g.set_entry_point("step_definer")
g.add_conditional_edges("step_definer", _router_branch,
    {"rag_execute": "rag_execute", "web_execute": "web_execute", END: END})
g.add_edge("rag_execute", "step_definer")   # loop back
g.add_edge("web_execute", "step_definer")   # loop back
```

This is a **loop graph**. After each execution step, control returns to `step_definer`. The loop exits when `_router_branch` returns `END`.

**Router logic**:
```python
def _router_branch(state: PlanExecState) -> str:
    current_idx = len(state["step_output"])      # how many steps done
    if current_idx >= len(state["plan"]):         # all steps done → exit
        return END
    step_type = state["step_question"][current_idx]["type"]
    return "rag_execute" if step_type == "retrieve_db" else "web_execute"
```

The router reads the *next* step's type from `step_question[current_idx]`. If the current idx already equals the plan length, all steps are done and the loop exits.

### 9.5 StepDefinerAgent

**File**: `agents/StepDefinerAgent.py`
**Type**: Synchronous node
**Dual role**: defines the next step OR produces the final summary

The agent checks `finished_step_id = len(step_output)` against `len(plan)`:

**Case A — Steps remain:**
- Takes `plan[finished_step_id]` as the current sub-question
- Builds a `memory_str` of all completed steps: `"Q: <task>, A: <answer>"` lines
- Calls LLM with `StepDefinerAgent.yaml` to produce a `StepTaskState`
- The memory context allows the LLM to refine the current sub-question using information from prior answers
- Returns `{"step_question": [StepTaskState], "token_usage": ...}`

**Case B — All steps done:**
- Loads `PlanSummarizerAgent.yaml` instead
- Calls LLM with all `step_notes` and the `original_question`
- Returns `{"stop": True, "plan_summary": synthesized_answer, "token_usage": ...}`

The dual-role design means the same node handles both step definition and final synthesis, keeping the loop graph simple.

**StepTaskState output**:
```python
{"task": "<precise task description>", "type": "retrieve_db" | "web_search"}
```

The `type` field determines which execution branch `_router_branch` selects.

### 9.6 Execution Nodes

**`single_rag_execute_node`** (synchronous):
```python
def single_rag_execute_node(state: PlanExecState) -> dict:
    next_idx = len(state["step_output"])
    rag_graph = create_single_rag_graph()      # retriever → extractor → QA
    result = rag_graph.invoke({"question": state["step_question"][next_idx], "token_usage": {}})
    raw = result["final_raw_answer"]
    answer = raw["answer"]
    return {
        "step_output": [{"task": state["step_question"][next_idx], "answer": answer}],
        "step_notes": result.get("notes", []),
        "token_usage": result["token_usage"]
    }
```

**`single_web_execute_node`** (async):
```python
async def single_web_execute_node(state: PlanExecState) -> dict:
    next_idx = len(state["step_output"])
    web_graph = await create_single_web_graph()    # web_search → extractor → QA
    result = await web_graph.ainvoke({"question": state["step_question"][next_idx], "token_usage": {}})
    ...
```

Both functions:
1. Determine which step to execute using `len(step_output)` as the current index
2. Create and invoke the appropriate inner subgraph
3. Append one item to `step_output` (the `operator.add` reducer handles accumulation)
4. Append notes to `step_notes`

### 9.7 Inner Subgraphs

**`create_single_rag_graph()`**:
```
START → retrieve → extract → answer → END
```
Uses `RagState`. Identical topology to the medium tier except it runs synchronously (`invoke` not `ainvoke`).

**`create_single_web_graph()`**:
```
START → web_search → extract → answer → END
```
Uses `RagState`. Web search result is passed to extract, then QA.

---

## 10. Shared Agents and Utilities

### 10.1 LLM Factory: `get_llm`

**File**: `agents/llm.py`

```python
def get_llm(temperature: float = 0, **kwargs) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,       # default: "gemini-2.0-flash"
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,       # default: 0 (deterministic)
        max_retries=5,
        **kwargs,
    )
```

All agents call `get_llm()` directly — there is no shared instance. Each call creates a new `ChatGoogleGenerativeAI` object pointing at the same API key. Temperature 0 ensures consistent, deterministic outputs.

The file also contains a commented-out alternative backend (Ollama/vLLM via `ChatOpenAI`) that can be activated by uncommenting, pointing at `http://localhost:11434/v1` for a local model like `qwen3.5:2b`.

### 10.2 Prompt Loader: `load_prompt.py`

```python
def load_yaml_prompt(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_system_prompt(path: str) -> str:
    return load_yaml_prompt(path)["messages"][0]["content"]

def get_user_prompt(path: str) -> str:
    return load_yaml_prompt(path)["messages"][1]["content"]
```

Every agent calls `get_system_prompt()` and `get_user_prompt()` at call time — the YAML file is re-read on every invocation. This means prompt changes take effect without restarting the server.

### 10.3 JSON Parsing Pattern

Every agent that expects structured JSON output uses the same defensive parsing pattern:

```python
content = response.content
if "```json" in content:
    content = content.split("```json")[1].split("```")[0].strip()
elif "```" in content:
    content = content.split("```")[1].split("```")[0].strip()

try:
    data = json.loads(content)
    result = SomeTypedDict(**data)
except Exception:
    result = SomeTypedDict(fallback_field=default_value)
```

This handles:
1. Clean JSON responses
2. JSON wrapped in ` ```json ` fences (common with Gemini)
3. JSON wrapped in plain ` ``` ` fences
4. Completely unparseable responses (fallback)

---

## 11. Prompt Engineering System

All prompts are stored as YAML files in `agents/Prompts/`. The format is a simple chat prompt with `_type`, `input_variables`, and a two-element `messages` list (system + user).

### 11.1 Prompt File Index

| File | Agent | Input Variables | Purpose |
|---|---|---|---|
| `DirectResponderAdvocate.yaml` | DirectResponderAdvocate | `question` | Vote easy/medium/complex based on general-knowledge criteria |
| `ContextualAnalystAdvocate.yaml` | ContextualAnalystAdvocate | `question` | Vote easy/medium/complex based on document-retrieval need |
| `DeepResearcherAdvocate.yaml` | DeepResearcherAdvocate | `question` | Vote easy/medium/complex based on multi-step need |
| `DirectResponderAgent.yaml` | DirectResponderAgent | `question` | Answer directly; set `web_needed` flag |
| `ExtractorAgent.yaml` | ExtractorAgent | `passage`, `query` | Extract relevant info as bullet points |
| `QuestionAnsweringAgent.yaml` | QuestionAnsweringAgent | `context`, `question` | Synthesize answer with confidence rating and sources |
| `PlannerAgent.yaml` | PlannerAgent | `question` | Decompose into ordered sub-questions |
| `StepDefinerAgent.yaml` | StepDefinerAgent | `plan`, `cur_step`, `memory` | Refine step + classify as retrieve_db or web_search |
| `PlanSummarizerAgent.yaml` | StepDefinerAgent (completion) | `step_notes`, `original_question` | Synthesize all step notes into a final answer |

### 11.2 Notable Prompt Design Details

**DirectResponderAgent**: Instructs model to provide a "partial answer if web_needed is true" — so even when web is needed, the initial pass provides what it can, and the web context is appended as supplementary material.

**StepDefinerAgent**: Receives `memory` — a formatted string of all completed `Q: ..., A: ...` pairs. This gives the LLM context about what has already been learned, allowing it to write a more precise task for the current step.

**ContextualAnalystAdvocate**: Emphasizes "single retrieval pass is sufficient" as the key criterion for medium — if chaining is required, it should vote complex.

**DeepResearcherAdvocate**: Explicitly lists "comparison, synthesis, trend analysis, or multi-entity reasoning" as complex criteria, ensuring genuinely research-heavy questions reach the complex tier.

---

## 12. External Integrations

### 12.1 Qdrant Vector Database

**Purpose**: Store and retrieve document embeddings for RAG.

**Configuration** (from `.env.dev`):
```
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
```

**Embedding Service**:
- URL: `http://{SERVER_HOST}:{EMBEDDING_PORT}/v1/embed`
- Default: `http://localhost:8080/v1/embed`
- Input: `{"query": "text"}`
- Output: `{"result": {"embedding": [float, ...]}}`
- Vector size: 1024 dimensions
- Distance metric: Cosine similarity

**Collection initialization**: The `Retriever.__init__()` checks if the collection exists and creates it if not, with `VectorParams(size=1024, distance=Distance.COSINE)`.

**Retrieval**: Top-5 results are returned. Each point payload contains `chunk_id` (string) and `page_content` (string).

### 12.2 Exa Web Search (via MCP)

**Purpose**: Real-time web search for current information.

**Integration**: `langchain_mcp_adapters.client.MultiServerMCPClient` connects to the Exa MCP server over `streamable_http` transport.

```python
client = MultiServerMCPClient({
    "exa": {
        "transport": "streamable_http",
        "url": f"{SMITHERY_EXA_URL}?api_key={EXA_MCP_API_KEY}&profile={EXA_MCP_PROFILE}"
    }
})
tools = await client.get_tools()
agent = create_react_agent(model=get_llm(), tools=tools)
```

The **ReAct agent** pattern means the LLM autonomously decides:
- Which Exa tools to call (search, find similar pages, etc.)
- What queries to form
- When to stop and return the answer

The `<think>...</think>` tag stripping handles reasoning model output (e.g., models that emit chain-of-thought wrapped in think tags before the final answer).

### 12.3 Google Gemini (via LangChain)

**Package**: `langchain_google_genai.ChatGoogleGenerativeAI`
**Model**: `gemini-2.0-flash` (default, configurable via `GEMINI_MODEL_NAME`)
**Temperature**: 0
**Max retries**: 5

All agents use the same `get_llm()` factory — no agent creates its own LLM instance directly.

---

## 13. API Layer and SSE Streaming

### 13.1 FastAPI Application

```python
# app.py
app = FastAPI(title="MultiAgentRAG API", version="1.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], ...)
```

CORS is configured for React dev servers at ports 5173 (Vite) and 3000 (CRA).

### 13.2 Endpoints

| Method | Path | Handler | Description |
|---|---|---|---|
| POST | `/api/chat` | `chat.py` | Streams SSE events for a query |
| GET | `/api/health` | `health.py` | Returns `{"status": "ok"}` |

The chat endpoint:
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_graph(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

`X-Accel-Buffering: no` disables Nginx buffering, ensuring tokens are delivered immediately.

### 13.3 ChatRequest Schema

```python
class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
```

Note: `history` is accepted but currently not passed into the graph state. The graph only uses `request.message`.

### 13.4 SSE Event Format

All events are serialized by `api/sse.py`:
```python
def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
```

**Event Types:**

| Event Type | Fields | When emitted |
|---|---|---|
| `agent_start` | `type`, `agent` | When a LangGraph node begins execution (`on_chain_start`) |
| `token` | `type`, `agent`, `content` | On each LLM token streamed (`on_chat_model_stream`) |
| `agent_end` | `type`, `agent` | When a node finishes (`on_chain_end`) |
| `routing` | `type`, `complexity` | When `discussion_panel` streams its result |
| `done` | `type`, `final_answer` | When the entire graph completes |
| `error` | `type`, `message` | On any unhandled exception |

### 13.5 Graph Streaming Engine

**File**: `graph/run_graph.py`

Uses `graph.astream_events(initial_state, version="v2")` — the LangGraph v2 streaming API that emits rich events for every node transition and LLM token.

Key logic:

```python
active_nodes: set[str] = set()   # prevents duplicate start/end events

async for event in graph.astream_events(initial_state, version="v2"):
    kind = event["event"]
    node_name = event["name"]
    agent_id = NODE_TO_AGENT.get(node_name, node_name)

    if kind == "on_chain_start" and node_name in NODE_TO_AGENT:
        if node_name not in active_nodes:
            active_nodes.add(node_name)
            yield sse({"type": "agent_start", "agent": agent_id})

    elif kind == "on_chain_stream" and node_name == "discussion_panel":
        complexity = chunk.get("complexity", "")
        if complexity:
            yield sse({"type": "routing", "complexity": complexity})

    elif kind == "on_chat_model_stream":
        # Find the parent agent by scanning event tags
        parent_agent = "AssistantAgent"
        for tag in event.get("tags", []):
            if tag in NODE_TO_AGENT:
                parent_agent = NODE_TO_AGENT[tag]
                break
        yield sse({"type": "token", "agent": parent_agent, "content": content})

    elif kind == "on_chain_end" and node_name in ("easy_tier", "medium_tier", "complex_tier"):
        final_answer = event["data"]["output"]["final_answer"]

yield sse({"type": "done", "final_answer": final_answer})
```

The `active_nodes` set ensures that if a node fires `on_chain_start` multiple times (e.g., nested graphs), only the first is reported to the frontend.

### 13.6 NODE_TO_AGENT Mapping

```python
NODE_TO_AGENT = {
    "discussion_panel":              "DiscussionPanel",
    "direct_responder_advocate":     "DirectResponderAdvocate",
    "contextual_analyst_advocate":   "ContextualAnalystAdvocate",
    "deep_researcher_advocate":      "DeepResearcherAdvocate",
    "moderator":                     "ModeratorAgent",
    "direct_responder":              "DirectResponderAgent",
    "web_search":                    "WebSearchAgent",
    "easy_tier":                     "EasyTier",
    "retrieve":                      "RetrieverAgent",
    "extract":                       "ExtractorAgent",
    "answer":                        "QuestionAnsweringAgent",
    "medium_tier":                   "MediumTier",
    "planner":                       "PlannerAgent",
    "executor":                      "StepDefinerAgent",
    "complex_tier":                  "ComplexTier",
    "rag_execute":                   "RetrieverAgent",
    "web_execute":                   "WebSearchAgent",
}
```

This mapping translates internal LangGraph node names into human-readable agent IDs. It is used both for SSE `agent_start`/`agent_end` events and for tagging streamed tokens with their parent agent.

---

## 14. Token Usage Tracking

Token tracking is a first-class concern throughout the system. Every LLM call extracts:
```python
token_usage = response.usage_metadata if hasattr(response, 'usage_metadata') else {}
```

`usage_metadata` from LangChain contains:
- `input_tokens` (or `prompt_tokens`)
- `output_tokens` (or `completion_tokens`)
- `total_tokens`

These dicts are returned by every agent and merged via the `sum_dicts` reducer as state flows through the graph. By the time the graph terminates, `state["token_usage"]` contains the **total tokens consumed across all agents and tiers** in the entire pipeline.

For `WebSearchAgent`, token usage is aggregated across all messages in the ReAct agent's conversation:
```python
for msg in messages:
    if isinstance(msg, AIMessage):
        if msg.usage_metadata:
            token_usage = sum_dicts(token_usage, msg.usage_metadata)
```

---

## 15. Data Flow Walkthroughs

### 15.1 Easy Tier: "What is the capital of France?"

```
POST /api/chat {"message": "What is the capital of France?"}
  │
  ▼
run_graph() → graph.astream_events(initial_state)
  │
  ▼
[discussion_panel]
  ├── DirectResponderAdvocate  → vote: "easy"   (general knowledge)
  ├── ContextualAnalystAdvocate → vote: "easy"  (no document retrieval needed)
  └── DeepResearcherAdvocate   → vote: "easy"   (single fact, no chaining)
  │
  ▼
[ModeratorAgent] → tally: {"easy": 3} → complexity = "easy"
  │
  ▼  SSE: {"type": "routing", "complexity": "easy"}
  │
  ▼
[easy_tier] → [direct_responder]
  │  LLM: "Paris"
  │  web_needed: false
  │
  ▼  (no web search)
  │
  ▼
SSE: {"type": "done", "final_answer": "Paris"}
```

### 15.2 Medium Tier: "What does the uploaded document say about Q3 revenue?"

```
[discussion_panel]
  ├── DirectResponderAdvocate  → vote: "medium" (references specific document)
  ├── ContextualAnalystAdvocate → vote: "medium" (single retrieval pass sufficient)
  └── DeepResearcherAdvocate   → vote: "easy"   (disagreement)
  │
  ▼
[ModeratorAgent] → {"medium": 2, "easy": 1} → complexity = "medium"
  │
  ▼
[medium_tier]
  ├── [retrieve] (parallel)
  │     embedding("Q3 revenue") → Qdrant cosine search → top-5 docs
  │     → [extract]: "Q3 revenue was $4.2M, up 12% YoY..."
  │
  └── [web_search] (parallel)
        ReAct agent → Exa search → web context (industry benchmarks)
  │
  ▼ (both converge)
  │
  ▼
[answer]
  context = notes + web_docs
  LLM: {"answer": "The document states Q3 revenue was $4.2M...",
         "confidence": "high", "sources": ["chunk_007"]}
  │
  ▼
SSE: {"type": "done", "final_answer": "The document states Q3 revenue was $4.2M..."}
```

### 15.3 Complex Tier: "Compare how AI is used in healthcare vs finance, and identify trends"

```
[discussion_panel] → complexity = "complex"

[complex_tier]
  ▼
[planner]
  plan = [
    "How is AI currently used in healthcare?",
    "How is AI currently used in finance?",
    "What are the latest trends in AI for healthcare?",
    "What are the latest trends in AI for finance?",
    "Compare the applications and trends across healthcare and finance"
  ]
  is_report = false

[complex_executor_node] → plan_executor_graph.ainvoke(...)

LOOP ITERATION 1:
  [step_definer]
    step 0: "How is AI used in healthcare?"
    type: "retrieve_db"  (internal knowledge base)
    → step_question[0] = {task: "...", type: "retrieve_db"}

  [rag_execute]
    create_single_rag_graph().invoke({"question": step_question[0]})
      → retrieve: top-5 healthcare AI docs from Qdrant
      → extract: bullet points of key facts
      → answer: "AI in healthcare includes diagnostics, drug discovery..."
    step_output[0] = {task: ..., answer: "AI in healthcare includes..."}
    step_notes += [extracted bullets]

LOOP ITERATION 2:
  [step_definer]
    memory = "Q: healthcare AI, A: diagnostics, drug discovery..."
    step 1: "How is AI used in finance?"
    type: "retrieve_db"
    → step_question[1] = {task: "...", type: "retrieve_db"}

  [rag_execute] → step_output[1], step_notes += [...]

LOOP ITERATION 3:
  [step_definer] → step 2: latest healthcare trends → type: "web_search"
  [web_execute] → ReAct agent → Exa → step_output[2], step_notes += [...]

LOOP ITERATION 4:
  [step_definer] → step 3: latest finance trends → type: "web_search"
  [web_execute] → step_output[3], step_notes += [...]

LOOP ITERATION 5:
  [step_definer] → step 4: compare → type: "retrieve_db"
  [rag_execute] → step_output[4], step_notes += [...]

FINAL ITERATION (step_output length == plan length):
  [step_definer]
    finished_step_id >= len(plan) → SUMMARY MODE
    Loads PlanSummarizerAgent.yaml
    LLM synthesizes: "AI applications in healthcare focus on diagnostics...
                      while finance prioritizes fraud detection...
                      Key trends include..."
    → stop = True, plan_summary = "..."

_router_branch returns END

complex_executor_node returns:
  {"final_answer": plan_summary, "token_usage": total_accumulated}

SSE: {"type": "done", "final_answer": "AI applications in healthcare..."}
```

---

## 16. Design Patterns and Architectural Decisions

### 16.1 Three-Advocate Voting Panel

**Pattern**: Panel/jury decision making
**Why**: No single agent reliably classifies query complexity. Using three agents with different perspectives and a majority vote makes the routing decision more robust. A 2-out-of-3 majority provides stronger signal than a single classifier.

**Trade-off**: Costs 3 LLM calls per query before any actual work begins. The staggered async delays add latency. However, misrouting (sending a simple query to the complex tier) would waste far more tokens.

### 16.2 Nested Subgraph Hierarchy

**Pattern**: Composite graphs
**Why**: Each tier is independently testable, compilable, and invocable. The medium tier graph can be invoked directly in tests without running the full main graph. The plan executor loop is separate from the complex tier top-level graph.

**Implementation**: Each subgraph is created fresh per invocation (`create_discussion_graph()` inside `discussion_panel_node`). This avoids stale state but pays a small compilation cost per call.

### 16.3 State Reducers for Fan-In and Accumulation

**Pattern**: Annotated fields with custom reducers
**Why**: LangGraph's default state merge overwrites fields. For use cases where multiple parallel nodes contribute to the same field, custom reducers are needed.

- `Annotated[List[AdvocateVote], operator.add]` → three parallel advocates each append one vote; state auto-concatenates
- `Annotated[dict, sum_dicts]` → any node contributing token usage gets summed, not overwritten
- `Annotated[List[StepOutput], operator.add]` → each loop iteration appends without overwriting previous steps

### 16.4 Index-Based Step Tracking (No Explicit Counter)

**Pattern**: Derive state from existing data
**Why**: The current step index is always `len(step_output)`. This is computed rather than stored, which prevents the counter from getting out of sync with the actual data. When the loop adds to `step_output`, the index automatically advances.

### 16.5 StepDefinerAgent as Both Executor and Synthesizer

**Pattern**: State-based role switching
**Why**: The plan executor loop only has one re-entrant node. By having `StepDefinerAgent` check whether all steps are complete and switch to synthesis mode, the loop graph remains simple (no extra terminal node needed). The exit condition is handled by the router (`_router_branch` returns `END`), but the actual synthesis happens in the same node that detects the completion condition.

### 16.6 Memory Context in StepDefiner

**Pattern**: Episodic memory as formatted string
**Why**: LLMs have no native memory between calls. By building `memory_str` = all prior `Q: ..., A: ...` pairs and including it in the prompt, each call to `StepDefinerAgent` has full context of what has been learned so far. This allows later steps to be more precise (e.g., "Given that X was found in step 2, search for Y specifically").

### 16.7 Defensive JSON Parsing

**Pattern**: Multiple fallback levels
**Why**: LLMs do not always produce perfectly formatted JSON. Gemini sometimes wraps output in markdown code fences. The three-level parsing strategy (clean JSON → strip fences → fallback to hardcoded default) ensures the system never crashes on a formatting quirk. The fallback values are chosen to be safe defaults that keep the system running.

### 16.8 Async with Staggered Delays

**Pattern**: Jittered request timing
**Why**: Three simultaneous LLM calls from the discussion panel can trigger 429 rate-limit errors on the Gemini API. Random jitter within defined ranges (not fixed sleeps) distributes the burst without adding predictable delay.

---

## 17. Configuration and Environment

All configuration is loaded from `.env.dev` via `python-dotenv`.

| Variable | Default | Used By |
|---|---|---|
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | `agents/llm.py` |
| `GOOGLE_API_KEY` | — | `agents/llm.py` |
| `QDRANT_PORT` | `6333` | `agents/retriever_node.py` |
| `QDRANT_COLLECTION_NAME` | `documents` | `agents/retriever_node.py` |
| `SERVER_HOST` | `localhost` | `agents/retriever_node.py` |
| `EMBEDDING_PORT` | `8080` | `agents/retriever_node.py` |
| `SMITHERY_EXA_URL` | — | `agents/WebSearchAgent.py` |
| `EXA_MCP_API_KEY` | — | `agents/WebSearchAgent.py` |
| `EXA_MCP_PROFILE` | — | `agents/WebSearchAgent.py` |

The LLM factory resolves the env file path relative to `agents/llm.py`:
```python
env_path = Path(__file__).resolve().parent.parent / ".env.dev"
load_dotenv(env_path)
```

---

## 18. Dependency Stack

From `pyproject.toml`:

| Package | Version | Role |
|---|---|---|
| `langgraph` | ≥0.6.10 | Graph orchestration, state management, streaming |
| `langchain` | ≥0.3.27 | LLM framework, message types |
| `langchain-google-genai` | ≥2.0.0 | Google Gemini integration |
| `langchain-mcp-adapters` | ≥0.1.11 | MCP client for Exa web search |
| `langchain-huggingface` | ≥0.3.1 | HuggingFace embedding support |
| `langchain-qdrant` | ≥0.2.1 | Qdrant integration layer |
| `qdrant-client` | ≥1.15.1 | Qdrant vector database client |
| `fastapi` | ≥0.115 | HTTP API framework |
| `uvicorn` | ≥0.30 | ASGI server |
| `sse-starlette` | ≥2.1 | Server-Sent Events response support |
| `pyyaml` | ≥6.0.3 | YAML prompt file parsing |
| `python-dotenv` | — | Environment variable loading |
| `torch` | 2.9.1 | ML framework (for embedding models) |

---

*Document generated: 2026-03-30*
