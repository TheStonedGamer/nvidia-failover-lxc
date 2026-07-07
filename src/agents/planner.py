import json
import traceback
from src.event_bus import EventBus, Event
from src.agents.base import BaseAgent
from src.llm import LocalLLMClient

PLANNER_SYSTEM_PROMPT = """You are the Lead Planner Agent for a local multi-agent engineering platform.
Your sole responsibility is to break a user software request down into a Directed Acyclic Graph (DAG) of sequential, verifiable tasks.

CRITICAL RULES:
1. Never write implementation code or code files in your tasks or planning.
2. Every task must be self-contained and have a concrete, programmatically verifiable condition (e.g. running a test, executing a build, checking compiler/LSP diagnostics, verifying file existence).
3. Identify dependencies between tasks clearly. If task B requires code or modules created in task A, task B MUST list task A in its dependencies.
4. Respond ONLY with a valid JSON object matching the schema below. Do not include markdown wraps (like ```json ... ```) or conversational prefix/suffix text in your raw response.

RESPONSE SCHEMA:
{
  "dag": [
    {
      "task_id": "unique-slug-1",
      "description": "Short, actionable description of what the Builder agent must implement.",
      "dependencies": [],
      "verifiable_condition": "A command or condition to verify the task (e.g., 'pytest tests/test_module.py')"
    },
    {
      "task_id": "unique-slug-2",
      "description": "Description of the next step.",
      "dependencies": ["unique-slug-1"],
      "verifiable_condition": "Diagnostic compile check or test script execution."
    }
  ]
}
"""

class PlannerAgent(BaseAgent):
    def __init__(self, event_bus: EventBus, llm_client: LocalLLMClient):
        super().__init__("planner", event_bus)
        self.llm_client = llm_client
        self.event_bus.subscribe("PLAN_REQUEST", self.handle_event)
        self.event_bus.subscribe("REPLAN_REQUEST", self.handle_event)

    async def handle_event(self, event: Event):
        if event.event_type == "PLAN_REQUEST":
            user_request = event.payload["request"]
            print(f"[Planner] Generating DAG for user request: '{user_request}'")
            
            user_prompt = f"User Request: {user_request}\n\nDecompose this request into a structured DAG."
            
            try:
                # Ask local LLM to generate the JSON DAG
                raw_response = await self.llm_client.chat_completion(
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_mode=True
                )
                
                # Clean up formatting anomalies (e.g. if the model wrapped it in markdown code blocks)
                cleaned = raw_response.strip()
                if cleaned.startswith("```"):
                    # Strip markdown wraps if present
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                
                parsed_dag = json.loads(cleaned)
                
                # Basic validation
                if "dag" not in parsed_dag or not isinstance(parsed_dag["dag"], list):
                    raise ValueError("Inference result did not contain a valid 'dag' array.")
                
                print(f"[Planner] Successfully generated DAG with {len(parsed_dag['dag'])} tasks.")
                await self.send_event(
                    event_type="DAG_PLANNED",
                    recipient="orchestrator",
                    payload={"dag": parsed_dag["dag"]}
                )
                
            except Exception as e:
                print(f"[Planner ERROR] Failed to generate or parse plan: {e}")
                traceback.print_exc()
                # Report failure to orchestrator
                await self.send_event(
                    event_type="PLAN_FAILED",
                    recipient="orchestrator",
                    payload={"reason": str(e)}
                )

        elif event.event_type == "REPLAN_REQUEST":
            failed_task_id = event.payload["failed_task_id"]
            reason = event.payload["reason"]
            diagnostics = event.payload.get("diagnostics", {})
            print(f"[Planner] Replanning triggered due to failure on task {failed_task_id}. Reason: {reason}")
            
            # Formulate replanning query
            user_prompt = (
                f"A task in the current execution DAG has failed.\n"
                f"Failed Task ID: {failed_task_id}\n"
                f"Failure Reason: {reason}\n"
                f"Diagnostics: {json.dumps(diagnostics)}\n\n"
                f"Please reconstruct a new corrected plan/DAG that avoids this failure."
            )
            
            try:
                raw_response = await self.llm_client.chat_completion(
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_mode=True
                )
                cleaned = raw_response.strip()
                # strip markdown wraps if present
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                    
                parsed_dag = json.loads(cleaned)
                print(f"[Planner] Successfully replanned DAG with {len(parsed_dag['dag'])} tasks.")
                await self.send_event(
                    event_type="DAG_PLANNED",
                    recipient="orchestrator",
                    payload={"dag": parsed_dag["dag"]}
                )
            except Exception as e:
                print(f"[Planner ERROR] Failed replanning: {e}")
                await self.send_event(
                    event_type="PLAN_FAILED",
                    recipient="orchestrator",
                    payload={"reason": f"Replanning failed: {e}"}
                )
