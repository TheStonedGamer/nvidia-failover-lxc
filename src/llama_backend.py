import os
import json
import subprocess
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import httpx

from src.ollama_resolver import resolver as ollama_resolver
from src.hardware import (
    find_llama_server,
    plan_gpu_layers,
    has_cuda_build,
    launch_env,
)


# ---------------------------------------------------------------------------
# Context Engine - pulls context from the Obsidian vault via obsidian-cli
# ---------------------------------------------------------------------------


class ContextEngine:
    """Provides context from Obsidian vault using obsidian-cli.

    The obsidian-cli `search` command returns one note path per line (plain text).
    This parses each line, resolves the note title, and fetches its content.
    """

    def __init__(self, vault_path: str = "C:\\Users\\BrianTheMint\\vault"):
        self.vault_path = vault_path

    async def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Search the vault and return concatenated note content for the top hits."""
        cmd = f'obsidian vault="vault" search query="{query}" limit={limit}'

        try:
            result = await asyncio.to_thread(self._run_obsidian_cli, cmd)

            if result.returncode == 0 and result.stdout:
                note_lines = [
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                ][:limit]

                context_parts = []
                for note_line in note_lines:
                    title = self._line_to_title(note_line)
                    content = await self.get_note_by_name(title)
                    if content:
                        context_parts.append(f"\n---\n# {title}\n\n{content}")

                return "\n".join(context_parts) if context_parts else ""
            return ""
        except Exception:
            return ""

    def _line_to_title(self, line: str) -> str:
        title = line
        if "/" in title:
            title = title.split("/")[-1]
        if "\\" in title:
            title = title.split("\\")[-1]
        if title.endswith(".md"):
            title = title[:-3]
        return title

    def _run_obsidian_cli(self, command: str) -> Any:
        try:
            return subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            return type(
                "obj",
                (object,),
                {"returncode": 1, "stdout": "", "stderr": "Command timed out"},
            )

    async def get_note_by_name(self, note_name: str) -> Optional[str]:
        cmd = f'obsidian vault="vault" read file="{note_name}"'
        try:
            result = await asyncio.to_thread(self._run_obsidian_cli, cmd)
            if result.returncode == 0 and result.stdout:
                return result.stdout
            return None
        except Exception:
            return None

    async def get_note_by_path(self, path: str) -> Optional[str]:
        full_path = os.path.join(self.vault_path, path)
        if not os.path.exists(full_path):
            return None
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Model Registry - tracks which expert models exist and which is loaded
# ---------------------------------------------------------------------------


@dataclass
class ExpertModel:
    """Definition of an expert model and how to launch it."""

    name: str  # internal alias, e.g. "builder-80b"
    gguf_path: str  # absolute path to the .gguf file
    role: str  # planner | builder | reviewer | general
    ctx_size: int = 32768
    n_gpu_layers: int = 99  # offload as many layers as fit
    n_cpu_threads: Optional[int] = None  # None = let llama.cpp decide
    description: str = ""
    # If set, this model is served on its own port (swapped in/out of VRAM).
    dedicated_port: Optional[int] = None


@dataclass
class ModelState:
    """Runtime state of an expert model."""

    expert: ExpertModel
    loaded: bool = False
    process: Optional[subprocess.Popen] = None
    port: Optional[int] = None
    last_used: float = 0.0


class ModelRegistry:
    """Registry and lifecycle manager for expert llama.cpp server processes.

    Architecture:
    - Orchestrator (qwen3-4b) runs permanently on CPU via a dedicated llama-server
      on port 8080. It is small enough to always stay resident.
    - Expert models (large, e.g. qwen3-coder-next-80b-a3e) are loaded into VRAM
      on demand. Only ONE expert is in VRAM at a time. Switching experts unloads
      the current one (kills its server process) and starts the new one.

    Experts can either:
      a) Share the orchestrator's port 8080 (sequential, simpler) - but this
         also kills the orchestrator during a swap, which we don't want.
      b) Run on their own dedicated ports (parallel lifecycle) - orchestrator
         stays up on 8080, expert swaps on 8081/8082/etc.

    We use approach (b): each expert has a dedicated_port. The proxy forwards
    to the orchestrator for routing decisions and to the expert's port for
    actual expert inference.
    """

    def __init__(self, orchestrator_port: int = 8080):
        self.orchestrator_port = orchestrator_port
        self.experts: Dict[str, ModelState] = {}
        self._next_expert_port = orchestrator_port + 1
        self._lock = asyncio.Lock()

    def register(self, expert: ExpertModel) -> None:
        """Register a new expert model definition."""
        if expert.dedicated_port is None:
            expert.dedicated_port = self._next_expert_port
            self._next_expert_port += 1
        self.experts[expert.name] = ModelState(expert=expert)

    def list_experts(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.expert.name,
                "role": s.expert.role,
                "loaded": s.loaded,
                "port": s.port,
                "gguf": s.expert.gguf_path,
            }
            for s in self.experts.values()
        ]

    def get_expert_for_role(self, role: str) -> Optional[ModelState]:
        """Find the registered expert for a given role."""
        for state in self.experts.values():
            if state.expert.role == role:
                return state
        return None

    def get_expert_by_name(self, name: str) -> Optional[ModelState]:
        return self.experts.get(name)

    async def ensure_expert_loaded(self, name_or_role: str) -> Optional[ModelState]:
        """Make sure the requested expert is loaded; swap out any other expert.

        Args:
            name_or_role: either an expert name ("builder-80b") or a role ("builder")
        Returns:
            The ModelState of the now-loaded expert, or None on failure.
        """
        async with self._lock:
            target = self.get_expert_by_name(name_or_role) or self.get_expert_for_role(
                name_or_role
            )
            if target is None:
                print(f"[Registry] No expert matches '{name_or_role}'")
                return None

            if target.loaded and target.process and target.process.poll() is None:
                target.last_used = asyncio.get_event_loop().time()
                return target

            # Unload any other currently-loaded expert to free VRAM
            for other_name, other_state in self.experts.items():
                if other_name == target.expert.name:
                    continue
                if other_state.loaded:
                    print(
                        f"[Registry] Swapping out '{other_name}' to free VRAM for '{target.expert.name}'"
                    )
                    await self._unload(other_state)

            # Load the target expert
            print(
                f"[Registry] Loading expert '{target.expert.name}' on port {target.expert.dedicated_port}"
            )
            ok = await self._load(target)
            if ok:
                target.last_used = asyncio.get_event_loop().time()
                return target
            return None

    async def _load(self, state: ModelState) -> bool:
        """Start a llama-server subprocess for the given expert."""
        expert = state.expert
        binary = find_llama_server()
        if not binary:
            print("[Registry] No llama-server binary found; cannot load expert.")
            state.loaded = False
            return False

        # Size GPU offload to the actual free VRAM. On a CPU-only build this
        # returns 0 (the flag is ignored anyway); on the CUDA build it offloads
        # all layers if the model fits, else a proportional partial offload.
        gpu_layers = plan_gpu_layers(expert.gguf_path, default=expert.n_gpu_layers)
        print(
            f"[Registry] Binary: {binary} (cuda={has_cuda_build()}); "
            f"n-gpu-layers={gpu_layers} for '{expert.name}'"
        )
        args = [
            binary,
            "--port",
            str(expert.dedicated_port),
            "--model",
            expert.gguf_path,
            "--ctx-size",
            str(expert.ctx_size),
            "--n-gpu-layers",
            str(gpu_layers),
        ]
        if expert.n_cpu_threads is not None:
            args += ["--threads", str(expert.n_cpu_threads)]

        print(f"[Registry] Launching: {' '.join(args)}")
        try:
            state.process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=launch_env(),
            )
        except Exception as e:
            print(f"[Registry] Failed to start llama-server for {expert.name}: {e}")
            state.loaded = False
            return False

        # Wait for the server to be ready (poll /v1/models)
        url = f"http://localhost:{expert.dedicated_port}/v1/models"
        async with httpx.AsyncClient() as client:
            for attempt in range(60):  # up to ~60s for large models
                try:
                    r = await client.get(url, timeout=2.0)
                    if r.status_code == 200:
                        state.loaded = True
                        state.port = expert.dedicated_port
                        print(
                            f"[Registry] Expert '{expert.name}' ready on port {state.port} (after {attempt}s)"
                        )
                        return True
                except Exception:
                    pass
                if state.process.poll() is not None:
                    print(
                        f"[Registry] llama-server for {expert.name} exited early with code {state.process.returncode}"
                    )
                    state.loaded = False
                    return False
                await asyncio.sleep(1)

        print(f"[Registry] Timeout waiting for {expert.name} to come up")
        await self._unload(state)
        return False

    async def _unload(self, state: ModelState) -> None:
        """Terminate an expert's llama-server subprocess to free VRAM."""
        if state.process and state.process.poll() is None:
            try:
                state.process.terminate()
                # wait up to 15s for graceful shutdown
                for _ in range(15):
                    if state.process.poll() is not None:
                        break
                    await asyncio.sleep(1)
                if state.process.poll() is None:
                    state.process.kill()
            except Exception as e:
                print(f"[Registry] Error stopping process: {e}")
        state.loaded = False
        state.process = None
        state.port = None

    async def unload_all(self) -> None:
        async with self._lock:
            for state in self.experts.values():
                await self._unload(state)


# ---------------------------------------------------------------------------
# MOE Orchestrator - uses the small router model to pick an expert
# ---------------------------------------------------------------------------


class MOEOrchestrator:
    """Mixture-of-Experts orchestrator.

    The orchestrator (small model on CPU) analyzes each request and decides
    which expert to dispatch to. Experts (large MoE models like qwen3-coder-
    next-80b-a3e) are loaded on demand into VRAM, one at a time.
    """

    def __init__(
        self,
        orchestrator_url: str = "http://localhost:8080/v1",
        orchestrator_model: str = "qwen3-4b",
        registry: Optional[ModelRegistry] = None,
    ):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.orchestrator_model = orchestrator_model
        self.context_engine = ContextEngine()
        self.registry = registry or ModelRegistry()

    # -- Expert routing ------------------------------------------------------

    async def route_request(
        self,
        messages: list,
        target_role: str = "orchestrator",
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Route a request either to the orchestrator or to an expert.

        For expert roles, we:
          1. Pull relevant context from the Obsidian vault.
          2. Ensure the expert model is loaded (swapping out any other expert).
          3. Forward the request to the expert's dedicated llama-server port.
        """
        # Orchestrator path - always available
        if target_role == "orchestrator":
            return await self._forward_to_url(
                self.orchestrator_url,
                self.orchestrator_model,
                messages,
                stream,
                temperature=0.3,
            )

        # Expert path - ensure expert is loaded, then forward to its dedicated port
        expert_state = await self.registry.ensure_expert_loaded(target_role)
        if expert_state is None or not expert_state.loaded:
            # Fallback to orchestrator if no expert available
            print(
                f"[MOE] No expert for role '{target_role}', falling back to orchestrator"
            )
            return await self._forward_to_url(
                self.orchestrator_url,
                self.orchestrator_model,
                messages,
                stream,
                temperature=0.1,
            )

        # Inject context from Obsidian vault
        messages = await self._inject_context(messages, target_role)

        return await self._forward_to_url(
            f"http://localhost:{expert_state.port}/v1",
            expert_state.expert.name,
            messages,
            stream,
            temperature=0.1,
        )

    async def _forward_to_url(
        self,
        base_url: str,
        model_name: str,
        messages: list,
        stream: bool,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": stream,
                    },
                    timeout=300.0,  # large experts can be slow on first token
                )
                if response.status_code == 200:
                    return response.json()
                return {
                    "error": f"LLM server returned {response.status_code}",
                    "status_code": response.status_code,
                    "model_used": model_name,
                }
            except httpx.RequestError as e:
                return {
                    "error": f"Failed to connect to LLM server: {e}",
                    "status_code": 503,
                    "model_used": model_name,
                }

    async def _inject_context(self, messages: list, role: str) -> list:
        """Prepend relevant Obsidian context to the system message."""
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        if not user_query:
            return messages
        context = await self.context_engine.get_relevant_context(user_query, limit=3)
        if not context:
            return messages
        messages = list(messages)  # don't mutate caller's list
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"]
                + f"\n\nYou are the {role} agent. Relevant project context from the vault:\n{context}\n---",
            }
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"You are the {role} agent. Relevant project context from the vault:\n{context}\n---",
                },
            )
        return messages

    # -- Orchestrator decision API -------------------------------------------

    async def orchestrate(
        self, user_request: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ask the small router model to decide which expert to dispatch to."""
        messages = [
            {"role": "system", "content": self._orchestrator_prompt()},
            {"role": "user", "content": user_request},
        ]
        if context:
            messages.append({"role": "context", "content": json.dumps(context)})
        return await self.route_request(messages, target_role="orchestrator")

    def _orchestrator_prompt(self) -> str:
        experts = self.registry.list_experts()
        expert_list = (
            "\n".join(f"  - '{e['name']}' (role: {e['role']})" for e in experts)
            or "  (no experts registered yet)"
        )
        return f"""You are the Orchestrator in a Mixture of Experts (MOE) system.

Available expert models:
{expert_list}

Your job:
1. Read the user's request.
2. Decide which expert should handle it. Return JSON:
   {{
     "decision": "orchestrate" | "route",
     "target_expert": "<expert name or role>",
     "instructions": "what the expert should do",
     "context_needed": ["list", "of", "files", "or", "topics"]
   }}

If the request is a simple question or chit-chat, use decision="orchestrate" and answer directly.
Otherwise use decision="route" with the most appropriate expert name.
"""

    # -- Model listing for the proxy /v1/models endpoint ---------------------

    async def get_models(self) -> List[Dict[str, str]]:
        """Combine orchestrator model with registered experts for /v1/models."""
        models = [{"id": self.orchestrator_model, "alias": "orchestrator"}]
        for e in self.registry.list_experts():
            models.append(
                {
                    "id": e["name"],
                    "alias": e["role"],
                    "loaded": str(e["loaded"]).lower(),
                    "port": str(e["port"]) if e["port"] else "",
                }
            )
        return models


# ---------------------------------------------------------------------------
# Default registry configuration
# ---------------------------------------------------------------------------


def _resolve_expert_gguf(role: str, default_tag: str) -> Optional[str]:
    """Resolve an expert's GGUF path for a role.

    Precedence:
      1. ``{ROLE}_MODEL_PATH`` env pointing at a raw .gguf file (bypass Ollama).
      2. ``ROUTER_{ROLE}_TAG`` env naming an Ollama tag, else ``default_tag``,
         resolved to its blob path via the Ollama store.
    Returns None if nothing resolves (the role is then simply not registered).
    """
    raw = os.environ.get(f"{role.upper()}_MODEL_PATH")
    if raw and os.path.exists(raw):
        return raw
    tag = os.environ.get(f"ROUTER_{role.upper()}_TAG", default_tag)
    return ollama_resolver.resolve_gguf(tag)


def build_default_registry() -> ModelRegistry:
    """Build a registry of expert models sourced from the Ollama store.

    Ollama is the model *catalog*; our custom llama.cpp binary is the *runtime*.
    Each expert is defined by an Ollama tag, resolved to the GGUF blob Ollama
    already stored on disk, and served by a dedicated llama-server on its own
    port so the orchestrator (8080) stays up during VRAM swaps.

    Defaults use the mid-size local models as a test ladder; override any role
    with ``ROUTER_<ROLE>_TAG`` (e.g. point builder at the 80B) or with a raw
    ``<ROLE>_MODEL_PATH`` gguf.
    """
    registry = ModelRegistry(orchestrator_port=8080)

    # role, expert name, default ollama tag, dedicated port, description
    expert_specs = [
        ("planner", "planner", "gemma4:26b", 8082, "Gemma 4 26B (A4B) planner"),
        ("builder", "builder", "qwen3-coder:30b", 8081, "Qwen3-Coder 30B builder"),
        (
            "reviewer",
            "reviewer",
            "nemotron-cascade-2:30b",
            8083,
            "Nemotron Cascade 2 30B (A3B) reviewer",
        ),
    ]

    for role, name, default_tag, port, description in expert_specs:
        gguf = _resolve_expert_gguf(role, default_tag)
        if gguf:
            registry.register(
                ExpertModel(
                    name=name,
                    gguf_path=gguf,
                    role=role,
                    ctx_size=32768,
                    n_gpu_layers=99,
                    description=description,
                    dedicated_port=port,
                )
            )
            print(f"[Registry] Registered {role} expert '{name}' -> {gguf}")
        else:
            print(
                f"[Registry] No gguf resolved for {role} (tag default: {default_tag}); skipping"
            )

    return registry


# Global instance - built lazily so missing models don't crash imports
moe_backend = MOEOrchestrator(registry=build_default_registry())
