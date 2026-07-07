from typing import Dict, List
from src.event_bus import EventBus, Event

class TaskTracker:
    def __init__(self, task_id: str, description: str, dependencies: List[str], verifiable_condition: str, budget: int = 3):
        self.task_id = task_id
        self.description = description
        self.dependencies = dependencies
        self.verifiable_condition = verifiable_condition
        self.budget = budget
        self.status = "todo"  # "todo", "executing", "verifying", "correcting", "completed", "failed"
        self.attempts = []


class LoopOrchestrator:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.task_registry: Dict[str, TaskTracker] = {}
        self.state = "IDLE"  # "IDLE", "PLANNING", "READY_TO_EXECUTE", "COMPLETED", "FAILED"
        
        # Subscribe to orchestration events
        self.event_bus.subscribe("REQUEST_SUBMITTED", self.on_request_submitted)
        self.event_bus.subscribe("PLAN_FAILED", self.on_plan_failed)
        self.event_bus.subscribe("DAG_PLANNED", self.on_dag_planned)
        self.event_bus.subscribe("TASK_COMPLETED", self.on_task_completed)
        self.event_bus.subscribe("TASK_VERIFIED", self.on_task_verified)
        self.event_bus.subscribe("TASK_REJECTED", self.on_task_rejected)

    @property
    def name(self) -> str:
        return "orchestrator"

    async def on_plan_failed(self, event: Event):
        self.state = "FAILED"
        print(f"[Orchestrator] Planning failed: {event.payload.get('reason')}")

    async def on_request_submitted(self, event: Event):
        """Triggers the Planner to build an execution DAG."""
        self.state = "PLANNING"
        print("[Orchestrator] Request received. Dispatching to Planner.")
        await self.event_bus.publish(Event(
            event_id=f"orch-plan-{event.event_id}",
            event_type="PLAN_REQUEST",
            sender="orchestrator",
            recipient="planner",
            payload={"request": event.payload["request"]}
        ))


    async def on_dag_planned(self, event: Event):
        """Ingests the planned DAG from the Planner and starts execution."""
        dag = event.payload["dag"]
        self.task_registry.clear()
        
        for task in dag:
            tracker = TaskTracker(
                task_id=task["task_id"],
                description=task["description"],
                dependencies=task["dependencies"],
                verifiable_condition=task["verifiable_condition"]
            )
            self.task_registry[tracker.task_id] = tracker
            
        self.state = "READY_TO_EXECUTE"
        print(f"[Orchestrator] DAG registered with {len(self.task_registry)} tasks. Starting execution loop.")
        await self.dispatch_eligible_tasks()

    async def dispatch_eligible_tasks(self):
        """Finds tasks with all dependencies satisfied and dispatches them."""
        if self.state not in ("READY_TO_EXECUTE", "EXECUTING"):
            return
            
        dispatched_any = False
        all_completed = True
        
        for task_id, tracker in self.task_registry.items():
            if tracker.status == "completed":
                continue
            all_completed = False
            
            if tracker.status in ("executing", "verifying", "correcting"):
                continue
                
            # Check dependencies
            deps_satisfied = all(
                self.task_registry[dep_id].status == "completed"
                for dep_id in tracker.dependencies
            )
            
            if deps_satisfied:
                tracker.status = "executing"
                self.state = "EXECUTING"
                dispatched_any = True
                print(f"[Orchestrator] Dispatching task {task_id} to Builder: {tracker.description}")
                await self.event_bus.publish(Event(
                    event_id=f"orch-exec-{task_id}",
                    event_type="TASK_DISPATCHED",
                    sender="orchestrator",
                    recipient="builder",
                    payload={
                        "task_id": tracker.task_id,
                        "description": tracker.description,
                        "context": {}
                    }
                ))
                
        if all_completed:
            self.state = "COMPLETED"
            print("[Orchestrator] All tasks completed successfully.")
            await self.event_bus.publish(Event(
                event_id="orch-session-complete",
                event_type="SESSION_COMPLETED",
                sender="orchestrator",
                recipient="all",
                payload={}
            ))
        elif not dispatched_any and not any(t.status in ("executing", "verifying", "correcting") for t in self.task_registry.values()):
            # Deadlock check
            self.state = "FAILED"
            print("[Orchestrator] Deadlock detected or task failed. Halting.")

    async def on_task_completed(self, event: Event):
        """Triggered when Builder has finished working on a task. Hands off to Reviewer."""
        task_id = event.payload["task_id"]
        tracker = self.task_registry[task_id]
        tracker.status = "verifying"
        
        print(f"[Orchestrator] Task {task_id} marked complete by Builder. Routing to Reviewer.")
        await self.event_bus.publish(Event(
            event_id=f"orch-verify-{task_id}",
            event_type="VERIFY_REQUEST",
            sender="orchestrator",
            recipient="reviewer",
            payload={
                "task_id": task_id,
                "changes": event.payload["changes"],
                "verifiable_condition": tracker.verifiable_condition
            }
        ))

    async def on_task_verified(self, event: Event):
        """Triggered when Reviewer approves the changes."""
        task_id = event.payload["task_id"]
        tracker = self.task_registry[task_id]
        tracker.status = "completed"
        
        print(f"[Orchestrator] Task {task_id} verified successfully.")
        await self.dispatch_eligible_tasks()

    async def on_task_rejected(self, event: Event):
        """Triggered when Reviewer rejects the changes."""
        task_id = event.payload["task_id"]
        tracker = self.task_registry[task_id]
        tracker.budget -= 1
        
        if tracker.budget > 0:
            tracker.status = "correcting"
            print(f"[Orchestrator] Task {task_id} rejected. Retries left: {tracker.budget}. Dispatching correction details.")
            await self.event_bus.publish(Event(
                event_id=f"orch-correct-{task_id}",
                event_type="TASK_DISPATCHED",
                sender="orchestrator",
                recipient="builder",
                payload={
                    "task_id": task_id,
                    "description": f"FIX ERROR: {event.payload['reason']}",
                    "context": {
                        "errors": event.payload.get("diagnostics", {}),
                        "previous_changes": event.payload.get("changes", [])
                    }
                }
            ))
        else:
            tracker.status = "failed"
            self.state = "FAILED"
            print(f"[Orchestrator] Task {task_id} failed: execution budget exhausted.")
            await self.event_bus.publish(Event(
                event_id=f"orch-replan-{task_id}",
                event_type="REPLAN_REQUEST",
                sender="orchestrator",
                recipient="planner",
                payload={
                    "failed_task_id": task_id,
                    "reason": event.payload["reason"],
                    "diagnostics": event.payload.get("diagnostics", {})
                }
            ))
