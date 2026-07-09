# Evaluation Plan: Token Quantity & Accuracy Across Systems and Inference Modes

## Objective

Compare **MA-RAG** and **DMA-RAG** on:
1. **Token output quantity** — total, prompt (input), and completion (output) tokens consumed per query
2. **Answer accuracy** — ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, G-eval, LLM-as-Judge

Evaluated under two inference modes:
- **LLM API** — cloud-hosted model (Gemini 2.5 Flash Lite for DMA-RAG and MA-RAG)
- **Local SLM** — locally-served small language model via Ollama or vLLM (Qwen 2.5 3B or similar)

This yields **4 evaluation configurations** (2 systems × 2 inference modes).

---

## Current State (What Already Exists)

### Test harness (`Evaluation/`)
- `evaluate.py` — orchestrates test runs, calls runners, scores answers
- `metrics.py` — ROUGE and BERTScore scoring
- `runners/ma_rag_runner.py` — wraps MA-RAG graph invocation
- `runners/multi_agent_rag_runner.py` — wraps DMA-RAG graph invocation
- `test_set.json` — 5 medical Q&A questions (q001–q005); **too small for reliable evaluation**

### Token tracking — current issues

**DMA-RAG:**
- Each agent reads `response.usage_metadata` from the LangChain response
- The `sum_dicts` reducer in `graph/state/GraphState.py` accumulates token dicts across all agents
- **Bug:** Gemini's `usage_metadata` uses keys `input_tokens`/`output_tokens`/`total_tokens`, but `multi_agent_rag_runner.py` maps to `prompt_tokens`/`completion_tokens` — the latter come back as 0 for all Gemini runs
- Ollama/vLLM via `ChatOpenAI` uses `prompt_tokens`/`completion_tokens` — different keys than Gemini

**MA-RAG:**
- The runner wraps the entire graph in `get_openai_callback()` from `langchain-community`
- `get_openai_callback()` only intercepts `ChatOpenAI` calls; **it does not capture tokens from `ChatGoogleGenerativeAI`** (`ExtractorAgent` uses Gemini)
- Mixed-provider token counts are therefore incomplete and misleading

### Model switching — current state

**DMA-RAG:** Has a single central factory `agents/llm.py`. Two implementations exist but one is commented out:
- Active: `ChatGoogleGenerativeAI` (reads `GEMINI_MODEL_NAME`, `GOOGLE_API_KEY` from `.env.dev`)
- Commented out: `ChatOpenAI` pointing at Ollama/vLLM (reads `VLLM_BASE_URL`, `VLLM_NAME`, `OPENAI_API_KEY`)

**MA-RAG:** No central factory. Each agent imports its own LLM client directly:
- `PlannerAgent`, `StepDefinerAgent`, `QuestionAnsweringAgent`, `WebSearchAgent` → `ChatOpenAI` with `VLLM_NAME`
- `ExtractorAgent` → `ChatGoogleGenerativeAI` with `GEMINI_MODEL`

---

## Phase 1 — Fix Token Tracking

### 1.1 DMA-RAG: normalize token key mapping

**File:** `DMA-RAG/agents/llm.py` and `Evaluation/runners/multi_agent_rag_runner.py`

The agent-level token extraction currently returns raw `usage_metadata` keys. These differ by provider:

| Provider | Input tokens key | Output tokens key | Total key |
|---|---|---|---|
| Gemini | `input_tokens` | `output_tokens` | `total_tokens` |
| Ollama/vLLM via ChatOpenAI | `prompt_tokens` | `completion_tokens` | `total_tokens` |

**Fix:** Create a normalizer helper (e.g., `agents/token_utils.py`) that maps both schemas to a single canonical form `{"input_tokens": N, "output_tokens": N, "total_tokens": N}`, called in every agent after `response.usage_metadata` is read.

Update `Evaluation/runners/multi_agent_rag_runner.py` to read canonical keys (`input_tokens`, `output_tokens`) instead of the incorrect `prompt_tokens`/`completion_tokens` mapping.

### 1.2 MA-RAG: replace `get_openai_callback` with per-call usage collection

**File:** `Evaluation/runners/ma_rag_runner.py`

`get_openai_callback()` silently drops Gemini tokens. Replace with direct usage collection:
- Patch or subclass `ExtractorAgent` to expose `usage_metadata` on its LangChain response
- Aggregate all agents' usage manually in the runner (or inject a callback that handles both providers)
- A simpler approach: add a `LangChainTracer`-based callback that records `usage_metadata` for every LLM call regardless of provider, accumulate in a list, then sum at the end

**Alternative (if MA-RAG stays in full-OpenAI mode for the local SLM run):** `get_openai_callback()` will correctly capture all calls when all agents use `ChatOpenAI`. Document that mixed-provider runs (OpenAI + Gemini) undercount tokens and must not be compared directly.

### 1.3 Add per-agent token breakdown (optional but valuable)

For DMA-RAG, each agent already returns `token_usage` in its state output. The runner could collect the per-agent breakdown (before `sum_dicts` collapses them) from intermediate graph events during streaming. This requires reading the SSE event stream from `run_graph.py` and tracking `agent_end` events that carry token data.

For the plan summary report, include:
- Total tokens per query
- Input vs output token split
- Per-agent breakdown for DMA-RAG (optional)

---

## Phase 2 — Model Switching Infrastructure

### 2.1 DMA-RAG: `LLM_BACKEND` env var in `agents/llm.py`

**File:** `DMA-RAG/agents/llm.py`

Replace the comment-based toggle with a runtime `LLM_BACKEND` switch:

```python
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")  # "gemini" | "local"

def get_llm(temperature: float = 0, **kwargs):
    if LLM_BACKEND == "local":
        return ChatOpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("VLLM_NAME", "qwen2.5:3b"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            temperature=temperature,
            **kwargs,
        )
    else:  # gemini
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_retries=5,
            **kwargs,
        )
```

Create two env files:
- `DMA-RAG/.env.gemini` — `LLM_BACKEND=gemini`, `GEMINI_MODEL_NAME=gemini-2.5-flash-lite`, `GOOGLE_API_KEY=...`
- `DMA-RAG/.env.local` — `LLM_BACKEND=local`, `VLLM_BASE_URL=http://localhost:11434/v1`, `VLLM_NAME=qwen2.5:3b`

### 2.2 MA-RAG: create central `agents/llm.py` factory

**Files to create/modify in `MA-RAG/agents/`:**

Create `MA-RAG/agents/llm.py`:
```python
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")  # MA-RAG was already local-by-default

def get_llm(temperature: float = 0, **kwargs):
    if LLM_BACKEND == "local":
        return ChatOpenAI(
            base_url=os.getenv("VLLM_BASE_URL", f"http://localhost:{os.getenv('VLLM_PORT', 8001)}/v1"),
            model=os.getenv("VLLM_NAME", "qwen2.5:3b"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            temperature=temperature,
            **kwargs,
        )
    else:  # gemini
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            **kwargs,
        )
```

Refactor each agent to import and call `get_llm()` instead of directly instantiating its own LLM client:
- `PlannerAgent.py` — replace `ChatOpenAI(model=os.getenv("VLLM_NAME"), ...)` with `get_llm()`
- `StepDefinerAgent.py` — same
- `QuestionAnsweringAgent.py` — same
- `WebSearchAgent.py` — same
- `ExtractorAgent.py` — replace `ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), ...)` with `get_llm()`

Create two env files:
- `MA-RAG/.env.gemini` — `LLM_BACKEND=gemini`, `GEMINI_MODEL_NAME=gemini-2.5-flash-lite`, `GOOGLE_API_KEY=...`
- `MA-RAG/.env.local` — `LLM_BACKEND=local`, `VLLM_BASE_URL=http://localhost:11434/v1`, `VLLM_NAME=qwen2.5:3b`

---

## Phase 3 — Evaluation Harness Updates

### 3.1 Add `--inference-mode` CLI argument to `evaluate.py`

**File:** `Evaluation/evaluate.py`

Add a new argument:
```
--inference-mode {gemini,local,both}   default: both
```

This will be passed through to each runner, which sets the appropriate environment for the system under test.

### 3.2 Update runner interface to accept inference mode

**Files:** `Evaluation/runners/ma_rag_runner.py`, `Evaluation/runners/multi_agent_rag_runner.py`

Each `run_*` function signature changes from:
```python
async def run_ma_rag(question: str) -> dict
```
to:
```python
async def run_ma_rag(question: str, inference_mode: str = "local") -> dict
```

The runner sets `os.environ["LLM_BACKEND"]` and the relevant model env vars before invoking the graph, based on the `inference_mode` argument. (The graphs read env vars at call time via `get_llm()`, so patching `os.environ` before invocation is sufficient.)

### 3.3 Update result schema

Add the following fields to each system result entry:
```json
{
  "inference_mode": "gemini" | "local",
  "model_name": "gemini-2.5-flash-lite" | "qwen2.5:3b",
  "token_usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  }
}
```

Update `print_summary_table` to group results by `(system, inference_mode)` and display a 4-row comparison table.

### 3.4 Expand the test set

**File:** `Evaluation/test_set.json`

5 questions is insufficient for statistically meaningful accuracy comparison. Target **at least 30 questions** (ideally 50–100) covering:
- Easy questions (factual, single-hop)
- Medium questions (require retrieval + synthesis)
- Complex questions (multi-step reasoning)

This provides enough data to:
- Compute confidence intervals on accuracy metrics
- Assess complexity-tier distribution in DMA-RAG
- Detect whether accuracy degrades meaningfully between LLM API and local SLM

Source additional questions from the same medical domain (consistent with existing q001–q005) or use a public medical QA dataset (MedQA, PubMedQA, BioASQ) as the source, writing ground truth answers based on the same document collection in Qdrant.

### 3.5 Output format for comparative analysis

Update the results JSON structure to support the 4-way comparison:

```json
{
  "metadata": {
    "timestamp": "...",
    "configurations": [
      {"system": "ma_rag",          "inference_mode": "gemini", "model": "gemini-2.5-flash-lite"},
      {"system": "ma_rag",          "inference_mode": "local",  "model": "qwen2.5:3b"},
      {"system": "multi_agent_rag", "inference_mode": "gemini", "model": "gemini-2.5-flash-lite"},
      {"system": "multi_agent_rag", "inference_mode": "local",  "model": "qwen2.5:3b"}
    ],
    "num_questions": 30
  },
  "results": [
    {
      "id": "q001",
      "question": "...",
      "ground_truth": "...",
      "system_results": {
        "ma_rag__gemini":          { "answer": "...", "token_usage": {...}, "latency_s": 0, "metrics": {...} },
        "ma_rag__local":           { "answer": "...", "token_usage": {...}, "latency_s": 0, "metrics": {...} },
        "multi_agent_rag__gemini": { "answer": "...", "token_usage": {...}, "latency_s": 0, "metrics": {...}, "complexity_tier": "medium" },
        "multi_agent_rag__local":  { "answer": "...", "token_usage": {...}, "latency_s": 0, "metrics": {...}, "complexity_tier": "medium" }
      }
    }
  ]
}
```

---

## Phase 4 — Local SLM Setup

### 4.1 Choosing the local model

The existing codebase references `qwen3.5:2b` (`.env`) and `qwen2.5:3b` (`.env.dev` alternative). For this evaluation:

- **Primary local SLM:** `qwen2.5:3b` via Ollama — small enough to run on consumer GPU/CPU, reasonable English instruction-following
- **Alternative:** `llama3.2:3b` or `phi3:mini` if Qwen2.5 underperforms

Both systems must use the **same local model** for a fair cross-system comparison.

### 4.2 Ollama setup

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull qwen2.5:3b

# Ollama serves at http://localhost:11434 by default
# Verify:
curl http://localhost:11434/api/tags
```

The OpenAI-compatible endpoint is at `http://localhost:11434/v1`, used by `ChatOpenAI(base_url=...)`.

### 4.3 vLLM alternative (higher throughput)

For batch evaluation, vLLM provides higher throughput than Ollama:
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8001 \
  --served-model-name qwen2.5:3b
```

Set `VLLM_BASE_URL=http://localhost:8001/v1` in the local env files.

### 4.4 Verify local inference before evaluation

Test that both systems can complete a single query with the local model before running the full harness:
```bash
cd Evaluation
uv run python -c "
import asyncio
from runners.multi_agent_rag_runner import run_multi_agent_rag
result = asyncio.run(run_multi_agent_rag('What is hypertension?', inference_mode='local'))
print(result)
"
```

---

## Phase 5 — Run Evaluations

### 5.1 Execution order

Run configurations **sequentially** to avoid resource contention (GPU memory, network rate limits):

```bash
cd Evaluation

# 1. MA-RAG with Gemini API
uv run python evaluate.py --system ma_rag --inference-mode gemini

# 2. MA-RAG with local SLM
uv run python evaluate.py --system ma_rag --inference-mode local

# 3. DMA-RAG with Gemini API
uv run python evaluate.py --system multi_agent_rag --inference-mode gemini

# 4. DMA-RAG with local SLM
uv run python evaluate.py --system multi_agent_rag --inference-mode local

# Or run all 4 in a single session (sequential per configuration):
uv run python evaluate.py --system both --inference-mode both
```

### 5.2 Rate limiting considerations

- **Gemini API:** Free tier has 15 RPM (requests per minute). For 30 questions × 2 systems, add a configurable `--request-delay-s` argument (default 4s) between questions.
- **Local SLM:** No rate limit, but latency per query is much higher. Disable BERTScore for faster iteration on local runs (`--no-bertscore`), then re-enable for the final comparison run.

---

## Phase 6 — Analysis and Reporting

### 6.1 Metrics to compare across all 4 configurations

| Metric | Description |
|---|---|
| `avg_rouge1_f1` | ROUGE-1 F1 — unigram overlap with ground truth |
| `avg_rouge2_f1` | ROUGE-2 F1 — bigram overlap |
| `avg_rougeL_f1` | ROUGE-L F1 — longest common subsequence |
| `avg_bert_f1` | BERTScore F1 — semantic similarity |
| `avg_input_tokens` | Mean input (prompt) tokens per query |
| `avg_output_tokens` | Mean output (completion) tokens per query |
| `avg_total_tokens` | Mean total tokens per query |
| `avg_latency_s` | Mean end-to-end latency in seconds |

### 6.2 Summary table format (print_summary_table output)

```
System              | Mode   | Model                  | ROUGE-1 | ROUGE-L | BERT-F1 | Input Tok | Output Tok | Total Tok | Latency(s)
--------------------|--------|------------------------|---------|---------|---------|-----------|------------|-----------|----------
MA-RAG              | gemini | gemini-2.5-flash-lite  |   0.XX  |   0.XX  |   0.XX  |      XXXX |       XXXX |      XXXX |     XX.X
MA-RAG              | local  | qwen2.5:3b             |   0.XX  |   0.XX  |   0.XX  |      XXXX |       XXXX |      XXXX |     XX.X
DMA-RAG       | gemini | gemini-2.5-flash-lite  |   0.XX  |   0.XX  |   0.XX  |      XXXX |       XXXX |      XXXX |     XX.X
DMA-RAG       | local  | qwen2.5:3b             |   0.XX  |   0.XX  |   0.XX  |      XXXX |       XXXX |      XXXX |     XX.X
```

### 6.3 Additional analysis script (`Evaluation/analyze_results.py`)

Write a separate script that:
1. Loads one or more result JSON files
2. Computes per-complexity-tier breakdowns for DMA-RAG (easy/medium/complex vs accuracy and tokens)
3. Computes accuracy degradation: `(LLM_API_score - local_SLM_score) / LLM_API_score` per system
4. Computes token efficiency: `accuracy / total_tokens` ratio as a cost-effectiveness proxy
5. Outputs a markdown-formatted report and optionally saves per-question CSV for further analysis

---

## Implementation Order (Summary)

| Step | File(s) | Change |
|---|---|---|
| 1 | `DMA-RAG/agents/llm.py` | Add `LLM_BACKEND` switch; keep both `ChatGoogleGenerativeAI` and `ChatOpenAI` paths |
| 2 | `DMA-RAG/agents/token_utils.py` (new) | Normalize `usage_metadata` key differences between Gemini and Ollama |
| 3 | All `DMA-RAG/agents/*.py` agents | Call `normalize_token_usage(response.usage_metadata)` |
| 4 | `DMA-RAG/.env.gemini`, `.env.local` (new) | Per-mode env config files |
| 5 | `MA-RAG/agents/llm.py` (new) | Central factory with `LLM_BACKEND` switch |
| 6 | `MA-RAG/agents/PlannerAgent.py`, `StepDefinerAgent.py`, `QuestionAnsweringAgent.py`, `WebSearchAgent.py`, `ExtractorAgent.py` | Replace inline LLM init with `get_llm()` call |
| 7 | `MA-RAG/.env.gemini`, `.env.local` (new) | Per-mode env config files |
| 8 | `Evaluation/runners/multi_agent_rag_runner.py` | Accept `inference_mode` param; fix token key mapping |
| 9 | `Evaluation/runners/ma_rag_runner.py` | Accept `inference_mode` param; fix Gemini token tracking |
| 10 | `Evaluation/evaluate.py` | Add `--inference-mode` arg; update result schema; update summary table |
| 11 | `Evaluation/test_set.json` | Expand to 30+ questions across easy/medium/complex difficulty |
| 12 | `Evaluation/analyze_results.py` (new) | Post-hoc comparative analysis and reporting |

---

## Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Local SLM produces shorter/longer outputs than LLM API, skewing token counts | Track input and output tokens separately; analyze distribution, not just mean |
| Local SLM may not follow system prompts reliably, producing malformed structured outputs (e.g., JSON from ModeratorAgent routing) | Test JSON parsing failure rate; add fallback parsing in `ModeratorAgent` and router nodes |
| `get_openai_callback` missing Gemini tokens (MA-RAG) | Phase 1.2 addresses this; validate by checking that Gemini token counts are non-zero in MA-RAG test runs |
| Gemini rate limits causing evaluation to stall | Add configurable `--request-delay-s` CLI argument to `evaluate.py` |
| Results not comparable if test set questions are too easy (all systems score ~1.0) or too hard (all score ~0.0) | After expanding test set, run a pilot with 5 questions per tier; adjust ground truth length/specificity accordingly |
| MA-RAG lacks complexity routing — all queries go through the same multi-step pipeline regardless of difficulty | Document this as a structural difference in the analysis; compare DMA-RAG per-tier vs MA-RAG overall |

---

## TODO List

### Phase 1 — Fix Token Tracking

- [x] **1.1** Create `DMA-RAG/agents/token_utils.py` with a `normalize_token_usage(usage_metadata: dict) -> dict` function that maps both Gemini keys (`input_tokens`, `output_tokens`) and OpenAI/Ollama keys (`prompt_tokens`, `completion_tokens`) to the canonical schema `{"input_tokens": N, "output_tokens": N, "total_tokens": N}`
- [x] **1.2** Update `DMA-RAG/agents/PlannerAgent.py` — replace raw `response.usage_metadata` with `normalize_token_usage(response.usage_metadata)`
- [x] **1.3** Update `DMA-RAG/agents/StepDefinerAgent.py` — same
- [x] **1.4** Update `DMA-RAG/agents/ExtractorAgent.py` — same
- [x] **1.5** Update `DMA-RAG/agents/QuestionAnsweringAgent.py` — same
- [x] **1.6** Update `DMA-RAG/agents/DirectResponderAgent.py` — same
- [x] **1.7** Update `DMA-RAG/agents/DirectResponderAdvocate.py` — same
- [x] **1.8** Update `DMA-RAG/agents/ContextualAnalystAdvocate.py` — same
- [x] **1.9** Update `DMA-RAG/agents/DeepResearcherAdvocate.py` — same
- [x] **1.10** Update `DMA-RAG/agents/WebSearchAgent.py` — same
- [x] **1.11** Update `Evaluation/runners/multi_agent_rag_runner.py` — read `input_tokens`/`output_tokens` from the final state instead of the currently broken `prompt_tokens`/`completion_tokens` mapping
- [x] **1.12** Fix `Evaluation/runners/ma_rag_runner.py` — replace `get_openai_callback()` with a provider-agnostic token aggregation approach (e.g., accumulate `response.usage_metadata` per LLM call via a custom LangChain callback that handles both `ChatOpenAI` and `ChatGoogleGenerativeAI`); ensure `ExtractorAgent`'s Gemini tokens are captured
- [ ] **1.13** Validate the fix: run a single question through each system and assert `input_tokens > 0` and `output_tokens > 0` in the returned dict for both Gemini and local modes

### Phase 2 — Model Switching Infrastructure

- [x] **2.1** Update `DMA-RAG/agents/llm.py` — add `LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")` branch; `"gemini"` path uses `ChatGoogleGenerativeAI`, `"local"` path uses `ChatOpenAI` pointing at Ollama/vLLM; remove the existing comment-based toggle
- [x] **2.2** Create `DMA-RAG/.env.gemini` — set `LLM_BACKEND=gemini`, `GEMINI_MODEL_NAME=gemini-2.5-flash-lite`, `GOOGLE_API_KEY=...`, plus existing Qdrant vars
- [x] **2.3** Create `DMA-RAG/.env.local` — set `LLM_BACKEND=local`, `VLLM_BASE_URL=http://localhost:11434/v1`, `VLLM_NAME=qwen3.5:2b` (model available on this machine), `OPENAI_API_KEY=ollama`, plus existing Qdrant vars
- [x] **2.4** Create `MA-RAG/agents/llm.py` — central factory with the same `LLM_BACKEND` switch: `"gemini"` → `ChatGoogleGenerativeAI`, `"local"` → `ChatOpenAI` with Ollama base URL
- [x] **2.5** Refactor `MA-RAG/agents/PlannerAgent.py` — remove inline `ChatOpenAI(model=os.getenv("VLLM_NAME"), ...)` instantiation; import and call `get_llm()` from the new factory
- [x] **2.6** Refactor `MA-RAG/agents/StepDefinerAgent.py` — same
- [x] **2.7** Refactor `MA-RAG/agents/QuestionAnsweringAgent.py` — same
- [x] **2.8** Refactor `MA-RAG/agents/WebSearchAgent.py` — same
- [x] **2.9** Refactor `MA-RAG/agents/ExtractorAgent.py` — replace `ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), ...)` with `get_llm()` so it participates in the backend switch
- [x] **2.10** Create `MA-RAG/.env.gemini` — same structure as DMA-RAG's `.env.gemini` plus MA-RAG-specific vars (`VLLM_PORT`, `QDRANT_PORT`, etc.)
- [x] **2.11** Create `MA-RAG/.env.local` — same structure as DMA-RAG's `.env.local` plus MA-RAG-specific vars
- [x] **2.12** Smoke-test both systems with both backends: run a single query in each of the 4 configurations and confirm no import or env errors (graph creation verified; full invocation depends on running services)

### Phase 3 — Evaluation Harness Updates

- [x] **3.1** Update `Evaluation/evaluate.py` — add `--inference-mode {gemini,local,both}` CLI argument (default: `both`)
- [x] **3.2** Update `Evaluation/evaluate.py` — add `--request-delay-s FLOAT` CLI argument (default: `4.0`) and insert `asyncio.sleep(delay)` between questions when hitting the Gemini API
- [x] **3.3** Update `Evaluation/runners/multi_agent_rag_runner.py` — add `inference_mode: str = "gemini"` parameter to `run_multi_agent_rag()`; loads `.env.{mode}` with `override=True` before invoking the graph
- [x] **3.4** Update `Evaluation/runners/ma_rag_runner.py` — add `inference_mode: str = "gemini"` parameter to `run_ma_rag()`; same env-patching approach with `_TokenUsageCallback`
- [x] **3.5** Update `Evaluation/evaluate.py` — pass `inference_mode` through to each runner call; key results as `f"{system}__{inference_mode}"` (e.g., `"ma_rag__gemini"`)
- [x] **3.6** Update the result JSON schema in `evaluate.py` — add `inference_mode` and `model_name` fields to each system result entry; use canonical token keys (`input_tokens`, `output_tokens`, `total_tokens`)
- [x] **3.7** Update `Evaluation/evaluate.py` `print_summary_table` — group rows by `(system, inference_mode)`; add `Input Tok` and `Output Tok` columns alongside `Total Tok`
- [x] **3.8** Update `Evaluation/metrics.py` — add `compute_geval(prediction, reference, question) -> dict` function using Gemini that scores correctness, completeness, and relevance; returns `{"geval_score": float}` (0–1 scale)
- [x] **3.9** Update `Evaluation/metrics.py` — add `compute_llm_judge(prediction, reference, question) -> dict` function with a structured 4-criterion rubric; returns `{"llm_judge_score": float, "llm_judge_reasoning": str}`
- [x] **3.10** Update `Evaluation/evaluate.py` `score_answer()` — calls `compute_geval` and `compute_llm_judge`; `--no-llm-judge` flag skips both
- [x] **3.11** Add `geval_score` and `llm_judge_score` to the summary table output
- [x] **3.12** Expand `Evaluation/test_set.json` — 30 questions: 8 easy, 12 medium, 10 complex; each has `"difficulty"` field

### Phase 4 — Local SLM Setup

- [x] **4.1** Verify Ollama is installed: `ollama --version` → `ollama version is 0.18.0`
- [x] **4.2** Model `qwen3.5:2b` already present on this machine (confirmed via `ollama list`); used instead of `qwen2.5:3b` since it is pre-installed and matches existing `.env.dev` config
- [x] **4.3** Ollama OpenAI-compatible endpoint confirmed running at `http://localhost:11434`
- [ ] **4.4** (Optional) vLLM not set up — Ollama is sufficient for this evaluation run
- [x] **4.5** Smoke test DMA-RAG with local backend: `create_main_graph()` succeeded, nodes verified
- [ ] **4.6** Full end-to-end invocation smoke test (requires Qdrant and Ollama to both be up simultaneously — run manually when both services are live)
- [x] **4.7** Fallback JSON parsing already in place in all DMA-RAG and MA-RAG agents (markdown fence stripping before `json.loads`)

### Phase 5 — Run Evaluations

- [ ] **5.1** Run `uv run python evaluate.py --system ma_rag --inference-mode gemini --no-llm-judge` as a quick sanity check (5 existing questions, no slow metrics)
- [ ] **5.2** Confirm the result JSON contains non-zero `input_tokens`, `output_tokens`, and accurate `rouge1_f1` values
- [ ] **5.3** Run the full 4-configuration evaluation (all 30+ questions, all metrics):
  - [ ] `uv run python evaluate.py --system ma_rag --inference-mode gemini`
  - [ ] `uv run python evaluate.py --system ma_rag --inference-mode local`
  - [ ] `uv run python evaluate.py --system multi_agent_rag --inference-mode gemini`
  - [ ] `uv run python evaluate.py --system multi_agent_rag --inference-mode local`
- [ ] **5.4** Verify all 4 result JSON files are saved to `Evaluation/results/` with the correct schema
- [ ] **5.5** Check for error rates per configuration — any `"error": non-null` entries indicate LLM or parsing failures; investigate and fix before treating results as final

### Phase 6 — Analysis and Reporting

- [x] **6.1** Create `Evaluation/analyze_results.py` — CLI script that accepts one or more result JSON file paths as arguments
- [x] **6.2** Implement per-configuration aggregate statistics: mean and std for `rouge1_f1`, `rougeL_f1`, `bert_f1`, `geval_score`, `llm_judge_score`, `input_tokens`, `output_tokens`, `total_tokens`, `latency_s`
- [x] **6.3** Implement accuracy degradation metric: for each system, compute `(gemini_score - local_score) / gemini_score` across all accuracy metrics; surface which system degrades least when switching to local SLM
- [x] **6.4** Implement token efficiency metric: `accuracy / total_tokens` per configuration as a cost-effectiveness proxy; report as a ranked table
- [x] **6.5** Implement per-difficulty-tier breakdown: group questions by `"difficulty"` field in test set; compute all metrics per tier per configuration; for DMA-RAG cross-reference against `complexity_tier` in results to validate routing accuracy
- [x] **6.6** Implement DMA-RAG routing analysis: compute what fraction of questions were routed to each tier (easy/medium/complex) under Gemini vs local; check whether tier distribution shifts between inference modes
- [x] **6.7** Output a markdown-formatted summary report to `Evaluation/results/report_{timestamp}.md`
- [x] **6.8** Output a per-question CSV to `Evaluation/results/per_question_{timestamp}.csv` with all metrics for each question × configuration combination (suitable for external analysis in Excel/Python notebooks)
