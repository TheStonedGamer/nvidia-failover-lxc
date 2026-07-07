import json
import re
from src.event_bus import EventBus, Event
from src.agents.base import BaseAgent
from src.llm import LocalLLMClient
from src.tools import WorkspaceTools
from src.agents.react_utils import build_bounded_prompt, truncate_observation

REVIEWER_SYSTEM_PROMPT = """You are the Reviewer Agent. Your job is to verify that modifications implemented by the Builder meet the task requirements and are error-free.

You have access to these workspace tools:
1. read_file(path: str)
2. execute_command(command: str)

INSTRUCTIONS:
You must run tools to compile, lint, or run tests to verify the code quality and correctness.
For each turn, output a Thought explaining what verification check to perform, followed by exactly ONE Action block call.

FORMAT:
Thought: <reasoning>
Action: <tool_name>(<arguments...>)

EXAMPLES:
Thought: I will run pytest to verify unit tests pass.
Action: execute_command(command="pytest tests/")

Once verification is complete, output your final Decision.
FORMAT:
Thought: I have evaluated the tests and diagnostics output.
Decision: approved

OR:
Thought: The validation checks failed.
Decision: rejected
Reason: <Describe error logs, stack traces, compiler errors, or failing code lines in full detail>

CRITICAL: Do not approve if compilation or test execution returned a non-zero exit code or failed diagnostics.
"""

class ReviewerAgent(BaseAgent):
    def __init__(self, event_bus: EventBus, llm_client: LocalLLMClient, tools: WorkspaceTools):
        super().__init__("reviewer", event_bus)
        self.llm_client = llm_client
        self.tools = tools
        self.event_bus.subscribe("VERIFY_REQUEST", self.handle_event)

    async def handle_event(self, event: Event):
        task_id = event.payload["task_id"]
        changes = event.payload["changes"]
        verifiable_condition = event.payload.get("verifiable_condition", "")
        
        print(f"[Reviewer] Verifying task {task_id} (Verifiable Condition: '{verifiable_condition}')")
        
        history = [
            f"Focal Task ID: {task_id}",
            f"Changes Made: {json.dumps(changes)}",
            f"Target Verifiable Condition: {verifiable_condition}"
        ]
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            user_prompt = build_bounded_prompt(
                history,
                f"\n\nIterative turn {iteration}/{max_iterations}. What check is needed next?",
            )
            
            try:
                response = await self.llm_client.chat_completion(
                    system_prompt=REVIEWER_SYSTEM_PROMPT,
                    user_prompt=user_prompt
                )
                history.append(response)
                
                # Check for final decision
                if "Decision:" in response:
                    decision_part = response.split("Decision:", 1)[1].strip().lower()
                    
                    if "approved" in decision_part:
                        print(f"[Reviewer] Task {task_id} approved.")
                        await self.send_event(
                            event_type="TASK_VERIFIED",
                            recipient="orchestrator",
                            payload={
                                "task_id": task_id,
                                "changes": changes
                            }
                        )
                        return
                    elif "rejected" in decision_part:
                        reason = "Validation checks failed."
                        if "Reason:" in response:
                            reason = response.split("Reason:", 1)[1].strip()
                        
                        print(f"[Reviewer] Task {task_id} rejected. Reason: {reason[:100]}...")
                        await self.send_event(
                            event_type="TASK_REJECTED",
                            recipient="orchestrator",
                            payload={
                                "task_id": task_id,
                                "status": "rejected",
                                "reason": reason,
                                "changes": changes,
                                "diagnostics": {"errors": [{"message": reason}]}
                            }
                        )
                        return
                
                # Parse action call
                action_match = re.search(r"Action:\s*(\w+)\((.*)\)", response)
                if not action_match:
                    raise ValueError("Response did not output a valid Action block format.")
                
                tool_name = action_match.group(1)
                tool_args_str = action_match.group(2)
                
                args_dict = {}
                for arg_pair in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+))', tool_args_str):
                    key = arg_pair.group(1)
                    val = arg_pair.group(2) or arg_pair.group(3) or int(arg_pair.group(4))
                    args_dict[key] = val
                
                # Execute tool
                print(f"[Reviewer] Action -> Calling {tool_name}({args_dict})")
                if tool_name == "read_file":
                    result = self.tools.read_file(args_dict.get("path", ""))
                elif tool_name == "execute_command":
                    result = self.tools.execute_command(args_dict.get("command", ""))
                else:
                    result = {"status": "error", "message": f"Unknown tool: '{tool_name}'"}
                
                obs_str = truncate_observation(f"Observation: {json.dumps(result)}")
                print(f"[Reviewer] Observation outcome status: {result.get('status')}")
                history.append(obs_str)
                
            except Exception as e:
                err_msg = f"[Reviewer ERROR] Loop iteration {iteration} failed: {e}"
                print(err_msg)
                history.append(f"Observation: Error executing tool or parsing response. Details: {e}")
                
        # Fallback rejection
        print("[Reviewer] Rejection triggered: iteration limit reached without approval decision.")
        await self.send_event(
            event_type="TASK_REJECTED",
            recipient="orchestrator",
            payload={
                "task_id": task_id,
                "status": "rejected",
                "reason": "Reviewer timed out during evaluation checks.",
                "changes": changes
            }
        )
