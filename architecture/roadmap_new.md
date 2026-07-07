# Local AI Agent Architecture Roadmap

## Mission

Build a self-improving local AI engineering platform that rivals commercial coding agents while remaining fully self-hosted, modular, and hardware-aware.

The system should optimize for:

- Reliable autonomous software engineering
- Efficient use of GPU, CPU, RAM, and context
- Long-running project memory
- Minimal token waste
- High-quality code generation
- Continuous learning from previous work
- Deterministic, verifiable execution

---

# Guiding Philosophy

The model should not remember everything.

The system should remember everything.

Models are interchangeable reasoning engines.

The platform is responsible for supplying exactly the information each model needs for the current task.

---

# Phase 1 — Stable Multi-Agent Foundation

Maintain dedicated agents for separate responsibilities.

## Planner

Responsibilities:

- Break work into verifiable tasks
- Never write implementation code
- Build execution plans
- Estimate dependencies
- Detect missing information

## Builder

Responsibilities:

- Perform edits
- Use tools immediately
- Verify changes
- Continue until complete
- Never stop after creating a plan

## Reviewer

Responsibilities:

- Independently verify changes
- Validate language server diagnostics
- Reject incorrect work
- Force retries when necessary

---

# Phase 2 — Software-Level MoE

Instead of relying on one giant model, create an intelligent routing layer.

Example routing:

- Router → lightweight model
- Planner → Qwen 14B
- Fast coding → GPT-OSS 20B
- Heavy coding → Qwen3 Coder Next
- Deep reasoning → larger reasoning model
- Verification → DeepSeek R1

The router should classify requests automatically and select the most appropriate specialist.

Routing decisions should consider:

- Task type
- Estimated complexity
- Current VRAM availability
- Context size
- Previous failures
- Latency targets

## Phase 2.5 — Unified OpenAI-Compatible Backend (COMPLETE ✅)

Use llama.cpp as the inference **runtime** behind a single OpenAI-compatible entry point, with **Ollama retained as the model catalog/manager**. (Earlier drafts said "replace Ollama"; the settled design instead splits the two roles — see Phase 2.5b.)

### Status: Complete — end-to-end verified 2026-07-06

**Current State:**

- ✅ Phase 1 scaffold complete: EventBus, LoopOrchestrator, Planner/Builder/Reviewer agents
- ✅ LocalLLMClient supports OpenAI-compatible API (defaults now point at llama.cpp `:8080`)
- ✅ api_server.py MOE proxy running on port 5001
- ✅ WorkspaceTools implemented for Builder/Reviewer agents
- ✅ Ollama fully replaced by llama.cpp (`qwen3-4b-Q4_K_M.gguf` orchestrator on `:8080`, 32k ctx)
- ✅ Role-based routing implemented (`_infer_target_role` maps model string → orchestrator/planner/builder/reviewer)
- ✅ On-demand VRAM swapping registry (`ModelRegistry`) — one expert resident at a time, dedicated ports
- ✅ Obsidian vault context injection per expert role (`ContextEngine`)
- ✅ Streaming + non-streaming both verified end-to-end through the proxy
- ✅ Full agent loop verified: Planner generates DAG → Builder ReAct → Reviewer

### Bugs fixed this session (2026-07-06)

- `src/llm.py`: `from typing import Optional, dict` → crashed every import (`dict` is not in `typing`). Removed.
- `src/llm.py` / `run_session.py`: default endpoints still pointed at Ollama (`:11434`, `qwen2.5-coder:7b`). Repointed to llama.cpp (`:8080`, `qwen3-4b`).
- ruff: removed unused `asyncio` import, dead f-strings, and unused agent locals in `run_session.py`.

### Phase 2.5b — Ollama as model catalog, llama.cpp as runtime (COMPLETE ✅)

Settled architecture: **Ollama manages the models** (download, storage, versioning in `E:\Ollama`); **our custom llama.cpp runs them.** An Ollama model's `application/vnd.ollama.image.model` layer is a plain GGUF blob, so llama.cpp loads it directly by path — no copy, no conversion, Ollama's blob dedup preserved.

- ✅ `src/ollama_resolver.py` — resolves an Ollama tag (e.g. `gemma4:26b`) to its GGUF blob path by reading the manifest; also recovers stored sampling params and lists the store.
- ✅ `build_default_registry()` now defines experts by Ollama tag (override via `ROUTER_<ROLE>_TAG` or raw `<ROLE>_MODEL_PATH`), resolved to blobs at startup.
- ✅ Default expert ladder: planner=`gemma4:26b`, builder=`qwen3-coder:30b`, reviewer=`nemotron-cascade-2:30b`; top rung = `Qwen3-Coder-Next-80b-A3B`.
- ✅ **Verified live:** registry launched our llama.cpp `llama-server` against an extensionless Ollama blob, served a completion, and unloaded cleanly (GGUF magic-byte detection + VRAM swap both work).

### Known limitations

- ✅ ~~Unbounded Builder/Reviewer context~~ — **fixed.** `src/agents/react_utils.py` bounds history (rolling window + observation truncation); overflow errors 9→0.
- ✅ ~~Small-model loop stalls~~ — **mitigated.** Builder now detects consecutive identical actions and nudges toward `Final Answer`.
- ✅ ~~No expert models on disk~~ — **resolved** via the Ollama catalog integration above.
- ✅ ~~No hardware-aware selection~~ — **done.** `src/hardware.py` queries free VRAM (`nvidia-smi`), parses each GGUF's block count, and sizes `--n-gpu-layers` to fit (partial offload when a model exceeds VRAM). Wired into `ModelRegistry._load`.
- ✅ ~~CPU-only inference~~ — **done.** llama.cpp rebuilt with CUDA (`build-cuda`, `-DGGML_CUDA=ON`, `sm_86`); `hardware.py` auto-selects the CUDA binary over the CPU build and injects CUDA 13's `bin\x64` runtime DLLs into the launch env. Verified: RTX 3060 detected, 4B orchestrator **45 tok/s** (vs ~13 with 4 slots / CPU).
- **12 GB VRAM budget tension.** A 32k-ctx 4B orchestrator uses ~8.6 GB with `--parallel 1`, leaving ~3.5 GB — not enough to co-resident a 30B expert. Resolution options: run the orchestrator at a smaller ctx, or on CPU, and give the GPU to whichever expert is swapped in. Decide during Phase 2.7 testing.
- **Orchestrator loads are slow at 32k ctx.** Client timeout raised 120s→300s; large planning calls can still be slow. Smaller orchestrator ctx or a faster router model would help.

### Why llama.cpp over Ollama?

- **Better control** - Direct access to GGUF models, quantization options, and runtime flags
- **Single endpoint** - OpenAI-compatible API built-in, no additional proxy layer needed for basic use
- **Resource efficiency** - Better memory management, custom kernels, and fine-grained VRAM control
- **Portability** - Single binary distribution, easier to deploy across machines

### Architecture Changes

1. **Llama.cpp as Primary Backend**
   - Run llama.cpp server with multiple models loaded (or use dynamic loading/unloading)
   - Point OpenCode configuration to `http://localhost:8080/v1` (llama.cpp default port)
   - Remove Ollama dependency entirely

2. **Intelligent Proxy Layer (Enhanced api_server.py)**
   - Single entry point at `/v1/chat/completions`
   - Intercept requests targeting agent names (`planner`, `builder`, `reviewer`)
   - Automatically route to appropriate specialized model based on:
     - Task type (from system prompt or request metadata)
     - Estimated complexity (token count heuristic)
     - Current hardware state (VRAM, load)
   - Return standard OpenAI-compatible responses

3. **OpenCode Configuration**

   ```json
   {
     "provider": {
       "model-router": {
         "options": {
           "baseURL": "http://localhost:5001/v1"
         }
       }
     }
   }
   ```
   - OpenCode connects to proxy port 5001
   - Proxy routes internally based on agent/task type

4. **Model Loading Strategy**
   - **Option A - Pre-loaded**: Load all models in llama.cpp server at startup
     - Pro: Fast routing, no load latency
     - Con: Higher VRAM footprint

   - **Option B - Dynamic Loading**: Use llama.cpp's model swapping via API
     - Pro: Lower memory usage
     - Con: 10-30s load time when switching models

### Implementation Priority

**Do this first** - it simplifies the entire architecture by:

- Eliminating Ollama dependency
- Creating a single endpoint for OpenCode to connect to
- Enabling clean routing logic in one place
- Making the system more portable and maintainable

---

## Phase 2.6 — Intelligent Proxy Implementation (MOSTLY COMPLETE ✅)

Implement intelligent routing in `api_server.py` to handle agent-based requests.

### Tasks

1. **Agent Detection Logic**
   - Parse incoming request `model` field
   - Detect agent names: `planner`, `builder`, `reviewer`
   - Route to appropriate specialized model

2. **Routing Algorithm**

   ```python
   def resolve_model(request):
       target = request.get("model")

       if target not in ("planner", "builder", "reviewer"):
           return target  # Direct model targeting

       task_type = analyze_task(request)
       if task_type == "coding":
           return select_coding_model()
       elif task_type == "planning":
           return select_planning_model()
       elif task_type == "verification":
           return select_verification_model()
   ```

3. **Hardware-Aware Selection**
   - Query GPU VRAM availability (NVML)
   - Check model VRAM requirements
   - Consider latency targets and previous failures

4. **Streaming Support**
   - Ensure proper streaming response forwarding
   - Handle chunked responses correctly

### Acceptance Criteria

- [x] OpenCode can call `model="builder"` and get routed to appropriate coding model *(routing implemented; falls back to orchestrator until an expert gguf is present)*
- [x] Proxy returns standard OpenAI-compatible responses
- [ ] Hardware state is checked before routing (skip models that exceed VRAM) *(NVML still TODO)*
- [x] Streaming requests work end-to-end

---

## Phase 2.7 — Harden the Agent Loop (Next Session)

The plumbing works; the loop is not yet robust. Priorities, highest-impact first:

1. **Bound the ReAct context.** Truncate or summarize tool observations before appending to history. Cap `read_file` observations, drop old turns, or keep a rolling window. This is the #1 blocker — it currently caps real tasks at a handful of turns.
2. **Stronger stop-conditions in the Builder.** Detect repeated identical actions and force a `Final Answer:` or fail fast, so small models don't spin the iteration budget.
3. **Stand up a real builder expert.** Register `qwen2.5-coder-7b-Q4_K_M.gguf` (already on disk) as the `builder` expert on its dedicated port and verify a live VRAM swap + context injection round-trip. Delete/re-fetch the broken `qwen2.5-coder-7b-Q8_0.gguf`.
4. ✅ **Hardware-aware selection (DONE).** `src/hardware.py` queries free VRAM via `nvidia-smi` and sizes GPU offload per model (partial offload when it won't fully fit); auto-selects the CUDA binary. Remaining: refuse/queue a swap when even partial offload + orchestrator won't fit.
5. **Orchestrator-driven routing.** Wire `MOEOrchestrator.orchestrate()` (the small model's route/answer JSON decision) into the request path so routing is content-based, not just keyed off the model-name string.

---

## Phase 2.8 — External Escalation & Subagent Providers (PLANNED)

Extend the router beyond local models: when local models can't finish a task,
escalate to an **external subagent** — first a headless **Claude Code CLI**
subagent (agentic, on your subscription), then the **NVIDIA hosted API**
(`integrate.api.nvidia.com`, metered). Escalation is **local-first and
failure-driven**: cloud is only spent after local has genuinely failed a few
times, or when you explicitly ask for it.

### Design decisions (settled 2026-07-06)

- **Claude tier** = headless `claude -p` on the existing Max/Pro subscription
  (not the pay-per-token API). Gate on CLI auth + a rate-limit cooldown window.
- **NVIDIA tier** = hosted NIM at `https://integrate.api.nvidia.com/v1`
  (OpenAI-compatible, `NVIDIA_API_KEY`). Gate on remaining credits (detect
  402/429 → cooldown).
- **Triggers** = (a) local failure / low confidence, and (b) explicit request.
  **No** upfront difficulty *router* (see "Difficulty as accelerator" below).
- **Policy** = local-first; spend external usage only on need. Prefer the
  cheaper tier: `local (free) → claude_cli (subscription) → nvidia (credits)`.

### Two provider *shapes* (they are not interchangeable)

| Provider | Shape | Best for | How it's used |
|---|---|---|---|
| Local (llama.cpp) | Chat model | everything, first | drives our own ReAct loop |
| Claude Code CLI | **Agentic subagent** | multi-file Builder work | hand it the failing task + `agent_workspace`; it reads/writes/runs tools itself and returns a result |
| NVIDIA API | Strong chat model | Planner / Reviewer / hard single-shot reasoning | drop-in stronger backend for our ReAct loop |

So on escalation, a **Builder** task prefers the **Claude CLI** subagent
(genuinely edits files); a **Planner/Reviewer** task can use **either** — NVIDIA
is often enough and cheaper to reach.

### Escalation state machine (per task)

```
attempt on local expert  ──fail──▶  retry local (up to LOCAL_MAX_ATTEMPTS)
        │ success                        │ still failing
        ▼                                ▼
     done                        escalate to next AVAILABLE tier
                                  (claude_cli, then nvidia)
                                         │ all tiers exhausted / unavailable
                                         ▼
                                  report failure with diagnostics
```

- `LOCAL_MAX_ATTEMPTS` default **3** ("fails a few times") — an easy task (low
  difficulty hint) may use fewer before escalating; a flagged-hard task escalates
  after 1 local failure instead of grinding all 3.
- **Explicit request** (`force_tier`, or an `@claude` / `@nvidia` tag in the
  request) skips straight to that tier.
- Escalation only picks a tier whose `is_available()` currently returns true.

### Difficulty as an accelerator, NOT a router

Difficulty is a **free static heuristic** (no extra model call): keyword signals
(`concurrency`, `migration`, `race condition`, `refactor across`…) plus the
**planner DAG shape** (fan-out width, dependency depth) it already produced. It
is used for exactly two things and never routes on its own:

1. **Retry budget** — how many local attempts before escalating.
2. **Tier ordering** — which external tier to try first once escalating.

Rationale: the failure trigger already scores difficulty with *ground truth*
(a reviewer rejection is the task proving it was hard), which beats any upfront
prediction; a predictive router would only waste external usage on false
positives. A learned/empirical version can come later once Phase 12 metrics
exist.

### Availability & usage gating

Each provider implements `is_available() -> (bool, reason)`, cached with a short
TTL so we don't probe on every request:

- **ClaudeCLIProvider** — `claude` binary present + authenticated; optimistic by
  default, but on a `429` / "usage limit" result it records the reset time and
  reports unavailable until then (cooldown persisted to disk so restarts respect
  it).
- **NvidiaProvider** — `NVIDIA_API_KEY` set; holds an ordered **model failover
  ladder** (`ROUTER_NVIDIA_MODELS`, default `qwen3.5-397b → deepseek-v4-pro →
  qwen3.5-122b → nemotron-3-super-120b → llama-3.3-70b`). A `429` (rate limit)
  or `404/410` (EOL) sidelines only *that* model and the same `complete()` call
  fails over to the next rung — so a task rate-limited mid-run keeps going on a
  different model. Account-level `401/403/402` short-circuits to a provider-wide
  cooldown (another model won't help). Provider reports unavailable only when the
  whole ladder is cooling.
- **LocalProvider** — available whenever the llama.cpp server(s) respond.

### Components to build

1. `src/providers/base.py` — `Provider` ABC (`name`, `shape`, `is_available()`,
   `complete()` for chat, `run_task()` for agentic), plus a `Usage` record.
2. `src/providers/local.py` — wraps existing `LocalLLMClient` / `MOEOrchestrator`.
3. `src/providers/claude_cli.py` — `claude -p "<prompt>" --output-format json
   --model <sonnet|opus> --max-turns N` in `agent_workspace`; parse JSON,
   detect/record rate limits.
4. `src/providers/nvidia.py` — httpx client to `integrate.api.nvidia.com/v1`,
   OpenAI-compatible, credit/429 cooldown.
5. `src/providers/router.py` — `ProviderRouter`: preference order, availability
   cache, escalation policy, difficulty hint, retry budgets.
6. `src/difficulty.py` — static heuristic score from task text + DAG.
7. Config `providers.toml` (+ env overrides) — enable flags, order, per-provider
   model, `LOCAL_MAX_ATTEMPTS`, cooldown TTLs, soft budgets.
8. Orchestrator integration — track `attempt`/`tier` per task; on `TASK_FAILED`
   or review `REJECTED`, apply the state machine and re-dispatch; agents ask the
   `ProviderRouter` for the backend matching the task's current tier.
9. Usage/metrics logging — every escalation and external call logged (feeds
   Phase 12 continuous-improvement metrics and the future learned difficulty
   model).

### Acceptance criteria

- [ ] Local-only tasks never touch a provider network call.
- [ ] A task that fails local `LOCAL_MAX_ATTEMPTS` times auto-escalates to the
      first available external tier and can succeed there.
- [ ] Claude CLI subagent runs headless, edits files in `agent_workspace`, and
      returns a structured result; a rate-limit response disables the tier until
      its reset time (verified without crashing the loop).
- [x] **DONE.** NVIDIA tier serves a completion via `NVIDIA_API_KEY`; missing key
      or spent credits cleanly marks it unavailable rather than erroring. A
      mid-run `429`/EOL on one model transparently fails over down the model
      ladder (verified live: primary → deepseek-v4-pro → llama-3.3-70b all served).
- [ ] `@claude` / `@nvidia` explicit tags bypass local and go straight to tier.
- [ ] All external calls appear in the usage log.

---

# Phase 3 — Context OS

Build a dedicated context management layer that exists outside the models.

Its responsibility is to construct the best possible prompt for every request.

## Responsibilities

Maintain:

- Active task state
- Project summaries
- Architecture documents
- API references
- Code ownership
- Dependency graphs
- Recent edits
- Open issues
- Previous solutions

The Context OS should retrieve only the information needed for the current task.

---

# Phase 4 — Memory Architecture

Split memory into multiple layers.

## Working Memory

Short-lived.

Contains:

- Current task
- Current files
- Diagnostics
- Active edits

## Project Memory

Long-lived.

Contains:

- Architecture
- APIs
- Coding conventions
- Design decisions
- Project documentation

Stored in Obsidian using wiki links.

## Long-Term Memory

Contains:

- Successful solutions
- Common workflows
- User preferences
- Lessons learned
- Reusable implementations

---

# Phase 5 — Retrieval Engine

Never dump an entire project into context.

Instead:

1. Index everything.
2. Retrieve only relevant knowledge.
3. Rank by relevance and recency.
4. Compress before insertion.

Support:

- semantic retrieval
- keyword retrieval
- dependency retrieval
- architecture retrieval

---

# Phase 6 — Context Compression

Completed conversations should become concise memory objects.

Preserve:

- decisions
- rationale
- implementation
- affected files
- follow-up work

Discard unnecessary conversational history.

---

# Phase 7 — Semantic Cache

Avoid repeating expensive reasoning.

Cache:

- architecture questions
- API lookups
- code explanations
- previous fixes
- planning results

Invalidate only when relevant files change.

---

# Phase 8 — Hierarchical Project Maps

Maintain a live map of every project.

Include:

- modules
- services
- APIs
- dependencies
- ownership
- entry points
- build system
- tests

The agent should consult the map before scanning the codebase.

---

# Phase 9 — Incremental Indexing

Never rebuild indexes unnecessarily.

Only re-index:

- modified files
- renamed files
- deleted files
- changed dependencies

Everything else should remain cached.

---

# Phase 10 — Agent-Specific Memory

Each agent maintains focused memory.

Planner remembers:

- architecture
- roadmap
- design decisions

Builder remembers:

- implementation details
- current edits
- tool results

Reviewer remembers:

- recurring bugs
- rejected patterns
- common failures

---

# Phase 11 — Adaptive Context Budgets

Allocate context dynamically.

Priority order:

1. Current task
2. Active files
3. Diagnostics
4. Architecture
5. Memory
6. Conversation history

Conversation should be the first thing compressed.

---

# Phase 12 — Continuous Improvement

Track metrics automatically.

Examples:

- retries required
- review failures
- successful fixes
- token usage
- execution time
- model latency
- cache hit rate
- retrieval quality
- context utilization

Use these metrics to improve routing decisions over time.

---

# Long-Term Goal

Create a self-hosted engineering platform where models become replaceable components.

The intelligence should reside in:

- orchestration
- memory
- retrieval
- verification
- context assembly
- adaptive routing

The result should behave like a highly capable engineering assistant while remaining efficient on consumer hardware, scalable to larger systems, and resilient to future model changes.
