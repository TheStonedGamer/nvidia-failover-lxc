# Session Handoff — model-router (2026-07-05)

## Current State

### What's Built (Phase 1 Complete ✅)

| Component        | Status  | Location                                            |
| ---------------- | ------- | --------------------------------------------------- |
| EventBus         | ✅ Done | `src/event_bus.py` - async pub/sub engine           |
| LoopOrchestrator | ✅ Done | `src/orchestrator.py` - DAG execution state machine |
| PlannerAgent     | ✅ Done | `src/agents/planner.py` - generates task DAGs       |
| BuilderAgent     | ✅ Done | `src/agents/builder.py` - executes tasks with tools |
| ReviewerAgent    | ✅ Done | `src/agents/reviewer.py` - verifies changes         |

### MOE Proxy System (Phase 2 Complete ✅)

| Component       | Status  | Location                                         |
| --------------- | ------- | ------------------------------------------------ |
| ContextEngine   | ✅ Done | `src/llama_backend.py` - Obsidian vault context  |
| MOEOrchestrator | ✅ Done | `src/llama_backend.py` - Model routing engine    |
| Proxy Server    | ✅ Done | `src/api_server.py` - OpenAI-compatible endpoint |

### What's Running

- **Proxy Server**: `api_server.py` on port 5001 (forwarding to llama.cpp)
- **llama.cpp Server**: Running on port 8080 with qwen3-4b model
- **Default LLM**: Configured via env vars (`LOCAL_LLM_URL`, `LOCAL_LLM_MODEL`)
- **Test Harness**: `run_session.py` - executes full agent loop

### Configuration

```bash
# Environment variables (optional, has defaults)
export LOCAL_LLM_URL="http://localhost:8080/v1"
export LOCAL_LLM_MODEL="qwen3-4b"

# Run the proxy server
python src/api_server.py  # port 5001

# Run a test session
python run_session.py
```

## What's Next (Phase 2.5 → 2.6)

### Priority: Replace Ollama with llama.cpp

**Why**: Better control, single endpoint, resource efficiency, portability.

**Steps**:

1. **Install & Start llama.cpp**

   ```bash
   # Download and build llama.cpp server binary (completed)
   ./llama-server --port 8080 \
     --model models/qwen3-4b-Q4_K_M.gguf \
     --ctx-size 32768 \
     --n-gpu-layers 99
   ```

2. **Update Environment**

   ```bash
   export LOCAL_LLM_URL="http://localhost:8080/v1"
   export LOCAL_LLM_MODEL="qwen3-4b"
   ```

3. **Enhance api_server.py** (see Phase 2.6 for details)
   - Add agent detection logic
   - Implement routing algorithm
   - Add hardware-aware model selection

### Files to Modify

| File                | Change                                                          |
| ------------------- | --------------------------------------------------------------- |
| `src/api_server.py` | Add intelligent routing logic in `/v1/chat/completions` handler |
| `opencode.json`     | Update baseURL if llama.cpp runs on different port              |

## Known Issues & Gotchas

1. **Missing import in builder.py**: Line 105 uses `os.path.exists` but `os` isn't imported

   ```python
   # Add at top of src/agents/builder.py
   import os
   ```

2. **Streaming not tested**: api_server.py has streaming support but hasn't been verified end-to-end

3. **No VRAM monitoring yet**: Hardware-aware routing requires NVML bindings (nvidia-ml-py)

## Quick Start Commands

```bash
# 1. Start llama.cpp server (qwen3-4b model on port 8080)
./llama-server --port 8080 --model models/qwen3-4b-Q4_K_M.gguf --ctx-size 32768

# 2. Run the proxy (optional, for intelligent routing)
python src/api_server.py

# 3. Test a session
python run_session.py
```

## Architecture Diagram

```
┌─────────────┐
│   OpenCode  │ (connects to port 5001 or 8080)
└──────┬──────┘
       │ HTTP /v1/chat/completions
       ▼
┌─────────────────────────────────────┐
│    api_server.py (port 5001)        │ ⏳ Enhance for routing
│  - Pass-through to downstream       │
│  - Agent detection (TODO)           │
│  - Model selection (TODO)           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    llama.cpp server (port 8080)     │ ⏳ Replace Ollama
│  - OpenAI-compatible API            │
│  - Multiple models loaded           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Local LLM (qwen3-4b)             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│    Agent System (port 5001 or 8080) │
│  - EventBus → Orchestrator          │
│  - Planner → DAG generation         │
│  - Builder → ReAct loop with tools  │
│  - Reviewer → Verification loop     │
└─────────────────────────────────────┘
```

## Session Checklist

- [x] Replace Ollama with llama.cpp server on port 8080 ✅ (qwen3-4b model)
- [x] Test OpenCode connects to proxy endpoint ✅ (proxy forwarding to llama.cpp)
- [x] Implement MOE orchestrator with context engine ✅
- [x] Verify streaming responses work end-to-end ✅ (2026-07-06, SSE through proxy)
- [x] Run `run_session.py` with llama.cpp backend ✅ (2026-07-06, full Planner→Builder→Reviewer loop runs)

## 2026-07-05 Update: MOE System Implementation ✅

**Changes**:

- Created `src/llama_backend.py` with `MOEOrchestrator` class for Mixture of Experts routing
- Added `ContextEngine` class using obsidian-cli to fetch context from vault
- Updated `api_server.py` to use MOE backend with intelligent model routing
- Model roles: orchestrator (qwen3-4b), planner, builder, reviewer experts

## Related Notes

- [[model-router-roadmap-updates]] — Phase 2.5 details
- [[model-router-unified-endpoint]] — Architecture overview
- [[local-agent/MEMORY]] — Integration notes

## 2026-07-05 Update: llama.cpp Setup Complete ✅

**Changes**:

- Downloaded and built llama.cpp server from source (Windows/MSVC)
- Model: `qwen3-4b-Q4_K_M.gguf` (2.5 GB, good quality/size balance)
- Server running on port 8080 with reasoning enabled
- Proxy server (`api_server.py`) configured to forward to llama.cpp
- Environment variables updated: `LOCAL_LLM_URL=http://localhost:8080/v1`, `LOCAL_LLM_MODEL=qwen3-4b`
