import os
import asyncio
from src.event_bus import EventBus, Event
from src.orchestrator import LoopOrchestrator
from src.llm import LocalLLMClient
from src.tools import WorkspaceTools
from src.agents.planner import PlannerAgent
from src.agents.builder import BuilderAgent
from src.agents.reviewer import ReviewerAgent

async def main():
    print("=== Starting Model Router Agent Scaffold ===")
    
    # Configure LLM Client (override default Ollama endpoint with env vars if needed)
    llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:8080/v1")
    llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen3-4b")
    
    print(f"[System] Initializing LLM client (URL: {llm_url}, Model: {llm_model})")
    llm_client = LocalLLMClient(base_url=llm_url, default_model=llm_model)
    
    # Initialize workspace tools. Agents write generated code into a dedicated
    # sandbox dir (overridable via AGENT_WORKSPACE) so their output never
    # pollutes this project's own source tree.
    workspace_root = os.environ.get(
        "AGENT_WORKSPACE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_workspace"),
    )
    os.makedirs(workspace_root, exist_ok=True)
    print(f"[System] Agent workspace: {workspace_root}")
    tools = WorkspaceTools(workspace_root=workspace_root)
    
    # Initialize the core components
    bus = EventBus()
    
    # Instantiate the agents. Each subscribes itself to the event bus in its
    # constructor, so we keep the references alive but don't call them directly.
    agents = [
        PlannerAgent(bus, llm_client),
        BuilderAgent(bus, llm_client, tools),
        ReviewerAgent(bus, llm_client, tools),
    ]
    print(f"[System] Registered {len(agents)} agents on the event bus.")
    
    # Instantiate the orchestrator
    orchestrator = LoopOrchestrator(bus)

    
    # Submit a request
    user_request = "Build an API service with DB schema models and auth middleware."
    print(f"\n[User] Request: {user_request}\n")
    
    # Publish the request to trigger the Planner
    await bus.publish(Event(
        event_id="user-request-001",
        event_type="REQUEST_SUBMITTED",
        sender="user",
        recipient="orchestrator",
        payload={"request": user_request}
    ))
    
    # Monitor orchestrator state until it finishes
    while orchestrator.state not in ("COMPLETED", "FAILED"):
        await asyncio.sleep(0.5)
        
    print(f"\n=== Execution Finished. Final Orchestrator State: {orchestrator.state} ===")

if __name__ == "__main__":
    asyncio.run(main())

