# Local AI Agent Platform: Model-Router & Context OS

This repository contains the architectural blueprints, refined roadmap, and design specifications for a self-improving, hardware-aware local AI engineering platform.

## Project Structure

- `architecture/`
  - [roadmap_refined.md](file:///E:/Projects/model-router/architecture/roadmap_refined.md) - A phase-by-phase refined roadmap detailing engineering challenges, concrete technologies, and implementation specifications.
  - [roadmap_new.md](file:///E:/Projects/model-router/architecture/roadmap_new.md) - Master roadmap with Phase 2.5 added for unified llama.cpp backend.
  - [routing_layer.md](file:///E:/Projects/model-router/architecture/routing_layer.md) - Deep dive into the Software-Level MoE (Mixture of Experts) router, hardware monitoring, and latency-aware routing.
  - [context_os.md](file:///E:/Projects/model-router/architecture/context_os.md) - The prompt construction, AST tracking, and dynamic context budget allocation engine.
  - [memory_retrieval.md](file:///E:/Projects/model-router/architecture/memory_retrieval.md) - Storage schemas, vector database integration, Obsidian project memory layout, and the semantic caching layer.
  - [agent_communication.md](file:///E:/Projects/model-router/architecture/agent_communication.md) - Protocol definitions for the multi-agent foundation (Planner, Builder, Reviewer) and deterministic verification loops.

## Memory Index

- [[model-router-roadmap-updates]] — Phase 2.5: unified llama.cpp backend replacing Ollama
- [[model-router-unified-endpoint]] — Single entry point architecture for OpenCode integration
- [[SESSION_HANDOFF]] — Current status and next session tasks (2026-07-05)

## Core Vision

A local software engineering assistant designed to operate efficiently on consumer hardware (e.g., 24GB VRAM GPU / 64GB System RAM) using small, highly-specialized local models (ranging from 1.5B to 32B parameters) coordinated by an out-of-model context orchestrator and routing engine.
