# Unified Endpoint Architecture

## Overview

The model-router uses a single OpenAI-compatible entry point that handles all routing internally. This simplifies integration with tools like OpenCode while maintaining the flexibility of a multi-model system.

```
┌─────────────┐
│   OpenCode  │
└──────┬──────┘
       │ HTTP /v1/chat/completions
       ▼
┌─────────────────────────────────────┐
│    Proxy Server (port 5001)         │
│  - Intercept requests               │
│  - Analyze task type                │
│  - Route to appropriate model       │
└──────┬──────────────────────────────┘
       │
       ├──► Qwen 2.5 Coder 7B (builder)
       ├──► DeepSeek R1 Distill 14B (planner/reviewer)
       ├──► Qwen 2.5 Coder 32B (heavy coding)
       └──► [Other loaded models]
```

## How It Works

### 1. OpenCode Connects to Single Endpoint

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

### 2. Proxy Intercepts and Routes

When OpenCode sends a request:

```python
# Incoming request to proxy
{
  "model": "builder",  # or "planner", "reviewer", or specific model name
  "messages": [...],
  "temperature": 0.1
}
```

The proxy:

1. Checks if `model` is an agent name (`planner`, `builder`, `reviewer`)
2. If yes, applies routing logic to select appropriate backend model
3. Forwards request to llama.cpp with selected model
4. Returns response in OpenAI format

### 3. Routing Logic

```python
def resolve_model(request):
    target = request.get("model")

    # Direct model targeting (bypasses router)
    if target not in ("planner", "builder", "reviewer"):
        return target

    # Agent routing based on task analysis
    task_type = analyze_task(request)

    if task_type == "coding":
        return select_coding_model()
    elif task_type == "planning":
        return select_planning_model()
    elif task_type == "verification":
        return select_verification_model()
```

## Benefits

| Benefit               | Explanation                                               |
| --------------------- | --------------------------------------------------------- |
| **Simplified Config** | OpenCode only needs one `baseURL`                         |
| **Clean Separation**  | Agent logic separate from inference engine                |
| **Flexible Backend**  | Swap llama.cpp/Ollama/vLLM without changing client config |
| **Automatic Routing** | Agents work like virtual models, transparently routed     |

## Implementation Notes

### Current State (`src/api_server.py`)

The existing proxy is a simple pass-through. To enable intelligent routing:

1. Add agent detection logic
2. Implement hardware-aware model selection
3. Add caching for routing decisions
4. Support streaming responses properly

### Llama.cpp Integration

```bash
# Start llama.cpp with multiple models
./llama-server \
  --port 8080 \
  --model models/qwen2.5-coder-7b.Q8_0.gguf \
  --model models/deepseek-r1-distill-qwen-14b.Q8_0.gguf \
  --model models/qwen2.5-coder-32b.Q4_K_M.gguf
```

Proxy forwards to `http://localhost:8080/v1/chat/completions`.

## Migration Path

1. **Phase 1**: Replace Ollama with llama.cpp as backend
2. **Phase 2**: Add intelligent routing to proxy
3. **Phase 3**: Implement hardware-aware selection
4. **Phase 4**: Optimize model loading strategy
