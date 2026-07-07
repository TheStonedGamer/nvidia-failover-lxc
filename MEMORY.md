# model-router — Memory Index

Tiered local coding agent platform with intelligent routing through a unified OpenAI-compatible endpoint.

- [[model-router-roadmap-updates]] — Phase 2.5: llama.cpp backend replacing Ollama
- [[model-router-unified-endpoint]] — Single entry point architecture for OpenCode integration
- [[SESSION_HANDOFF]] — Current state and next session tasks

## Key Files

- `architecture/roadmap_new.md` - Master roadmap with Phase 2.6 (next session)
- `SESSION_HANDOFF.md` — This session's status and handoff notes
- `architecture/unified_endpoint.md` — Unified endpoint design document
- `opencode.json` - OpenCode configuration pointing to proxy at localhost:5001

## Current Status (2026-07-05)

**Phase 1 Complete**: EventBus, LoopOrchestrator, Planner/Builder/Reviewer agents all working.

**Phase 2.5 In Progress**: llama.cpp backend setup; api_server.py proxy ready but needs intelligent routing.

**Next Session Priority**: Implement agent detection and routing in `api_server.py`.

## Architecture Highlights

| Component                    | Description                                             |
| ---------------------------- | ------------------------------------------------------- |
| Proxy Layer (port 5001)      | Pass-through to downstream server (enhance for routing) |
| Llama.cpp Server (port 8080) | Target backend with OpenAI-compatible API               |
| Agent System                 | Planner→DAG, Builder→ReAct, Reviewer→Verify loop        |

## Migration Status

- [x] Phase 1 scaffold complete
- [x] LocalLLMClient supports OpenAI-compatible API
- [x] WorkspaceTools implemented for agents
- [ ] Replace Ollama with llama.cpp backend
- [ ] Add intelligent routing logic to api_server.py
- [ ] Implement hardware-aware model selection
