# Software-Level Mixture of Experts (MoE) Router

The router is the intelligent layer that selects the most suitable model for a given sub-task. It guarantees optimal performance, latency, and token efficiency while working within consumer hardware limits.

---

## Routing Input Schema

For every agent invocation, the orchestrator passes a routing query containing the task profile and hardware state:

```json
{
  "task": {
    "agent_type": "builder",
    "complexity": "high",
    "estimated_tokens_input": 12500,
    "target_latency_seconds": 15,
    "retry_count": 1
  },
  "hardware": {
    "available_vram_mb": 14200,
    "total_vram_mb": 24576,
    "system_ram_free_mb": 32110,
    "gpu_load_percent": 12
  },
  "history": {
    "failures_on_current_task": 0,
    "last_failed_model": null
  }
}
```

---

## Router Model Profiles

The system maintains a registry of local and remote models, indexed by their capabilities, VRAM footprints, and loading overheads.

```json
{
  "models": [
    {
      "id": "qwen-2.5-coder-1.5b",
      "displayName": "Qwen 2.5 Coder 1.5B (Q8)",
      "type": "local",
      "vram_required_mb": 2200,
      "context_window": 32768,
      "capabilities": ["classification", "fast_coding", "simple_fixes"],
      "average_tps": 85.0
    },
    {
      "id": "qwen-2.5-coder-7b",
      "displayName": "Qwen 2.5 Coder 7B (Q8)",
      "type": "local",
      "vram_required_mb": 8500,
      "context_window": 32768,
      "capabilities": ["coding", "ast_analysis", "minor_refactoring"],
      "average_tps": 45.0
    },
    {
      "id": "qwen-2.5-coder-32b",
      "displayName": "Qwen 2.5 Coder 32B (Q4_K_M)",
      "type": "local",
      "vram_required_mb": 20000,
      "context_window": 32768,
      "capabilities": ["planning", "complex_refactoring", "heavy_coding"],
      "average_tps": 22.0
    },
    {
      "id": "deepseek-r1-distill-qwen-14b",
      "displayName": "DeepSeek R1 Distill Qwen 14B (Q8)",
      "type": "local",
      "vram_required_mb": 16500,
      "context_window": 65536,
      "capabilities": ["reasoning", "complex_planning", "verification"],
      "average_tps": 18.0
    },
    {
      "id": "gpt-4o-mini",
      "displayName": "GPT-4o Mini (Cloud)",
      "type": "remote",
      "vram_required_mb": 0,
      "context_window": 128000,
      "capabilities": ["fallback", "planning", "fast_coding"],
      "average_tps": 120.0
    }
  ]
}
```

---

## Routing Selection Algorithm

The router selects models based on a weighted scoring function, taking hardware constraints and task requirements into account.

```python
def select_model(task_profile, hardware_state, model_registry):
    eligible_models = []

    for model in model_registry:
        # Check absolute constraints
        if model.type == "local" and model.vram_required_mb > hardware_state.available_vram_mb:
            continue  # Out of VRAM, skip

        if task_profile.estimated_tokens_input > model.context_window:
            continue  # Exceeds context window, skip

        if task_profile.agent_type == "reviewer" and "verification" not in model.capabilities and "reasoning" not in model.capabilities:
            # Low capability model for complex verification is not allowed
            if task_profile.complexity == "high" and model.id == "qwen-2.5-coder-1.5b":
                continue

        eligible_models.append(model)

    if not eligible_models:
        # Fallback to cloud API if local resources are completely saturated
        return get_cloud_fallback_model(model_registry)

    # Score eligible models based on complexity, current failures, and latency
    best_model = None
    highest_score = -1.0

    for model in eligible_models:
        score = 0.0

        # Capability Match
        if task_profile.agent_type == "builder" and "coding" in model.capabilities:
            score += 3.0
        elif task_profile.agent_type == "planner" and "planning" in model.capabilities:
            score += 3.0
        elif task_profile.agent_type == "reviewer" and "verification" in model.capabilities:
            score += 4.0

        # Reasoning weight for deep tasks
        if task_profile.complexity == "high" and "reasoning" in model.capabilities:
            score += 5.0

        # Latency constraint matching
        estimated_time = task_profile.estimated_tokens_input / model.average_tps
        if estimated_time <= task_profile.target_latency_seconds:
            score += 2.0

        # Avoid repeat failures
        if task_profile.last_failed_model == model.id:
            score -= 10.0

        if score > highest_score:
            highest_score = score
            best_model = model

    return best_model
```

---

## Dynamic Offloading & VRAM Management

When working with local servers like Ollama or Llama.cpp, loading and unloading models can introduce latency spikes (up to 15-30s depending on SSD read speeds and PCIe bandwidth).

### Mitigation Strategy

1.  **Model Pre-loading**: Keep the primary builder model (`qwen-2.5-coder-7b`) resident in VRAM permanently.
2.  **Context Shifting**: Enable `prompt_cache` features in Llama.cpp / Ollama to prevent recalculation of system prompts and core files.
3.  **Active Unload Triggers**: If a heavy reasoning model (e.g. `deepseek-r1-distill-qwen-14b`) is required, issue an unload command to other large models via the local runtime API before executing.

### Llama.cpp Specific Commands

```bash
# Start llama.cpp server with multiple models pre-loaded
./llama-server --port 8080 \
  --model models/qwen2.5-coder-7b.Q8_0.gguf \
  --model models/deepseek-r1-distill-qwen-14b.Q8_0.gguf \
  --ctx-size 32768 \
  --n-gpu-layers 99

# Or load/unload models dynamically via API
curl http://localhost:8080/v1/models -X POST \
  -H "Content-Type: application/json" \
  -d '{"model": "models/qwen2.5-coder-32b.Q4_K_M.gguf"}'
```

### Model Loading Strategies

| Strategy              | Pros                             | Cons                  | Best For                     |
| --------------------- | -------------------------------- | --------------------- | ---------------------------- |
| Pre-loaded all models | Instant routing, no load latency | High VRAM usage       | Workstations with 24GB+ VRAM |
| Dynamic loading       | Low memory footprint             | 10-30s switch latency | Laptop/consumer hardware     |
| Hybrid approach       | Balance of speed/memory          | More complex logic    | Most users                   |
