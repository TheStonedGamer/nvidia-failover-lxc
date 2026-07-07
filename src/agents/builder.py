import json
import re
import hashlib
import os
from src.event_bus import EventBus, Event
from src.agents.base import BaseAgent
from src.llm import LocalLLMClient
from src.tools import WorkspaceTools
from src.agents.react_utils import build_bounded_prompt, truncate_observation

BUILDER_SYSTEM_PROMPT = """You are the Builder Agent. Your job is to make codebase modifications and implement features to satisfy a given programming task.

You have access to these workspace tools:
1. read_file(path: str)
2. write_file(path: str, content: str)
3. grep_search(pattern: str, directory: str = ".")
4. execute_command(command: str)

INSTRUCTIONS:
You must run a Thought-Action-Observation loop to perform your work.
For each turn, output a Thought explaining your reasoning, followed by exactly ONE Action block call.

FORMAT:
Thought: <your reasoning>
Action: <tool_name>(<arguments...>)

EXAMPLES:
Thought: I need to check the current db schema before modifying it.
Action: read_file(path="src/models/schema.py")

Thought: I will compile the changes to see if they are valid.
Action: execute_command(command="python -m py_compile src/models/schema.py")

Once you have completed the task and verified your changes (using compile/test commands), output your final results matching the format below.
FORMAT:
Thought: I have implemented and verified all modifications.
Final Answer: {
  "changes": [
    {
      "file_path": "relative/path/to/file.py",
      "type": "NEW",
      "diff": "..."
    }
  ]
}

CRITICAL: Never output markdown wrappers (such as ```json) inside your Final Answer. Output only raw JSON after the 'Final Answer:' prefix.
"""


class BuilderAgent(BaseAgent):
    def __init__(
        self, event_bus: EventBus, llm_client: LocalLLMClient, tools: WorkspaceTools
    ):
        super().__init__("builder", event_bus)
        self.llm_client = llm_client
        self.tools = tools
        self.event_bus.subscribe("TASK_DISPATCHED", self.handle_event)

    async def handle_event(self, event: Event):
        task_id = event.payload["task_id"]
        description = event.payload["description"]
        context = event.payload.get("context", {})

        print(f"[Builder] Processing task {task_id}: '{description}'")

        # Prepare context history for the ReAct execution
        history = [
            f"Active Task Description: {description}",
            f"Previous Failures/Diagnostics: {json.dumps(context.get('errors', {}))}",
        ]

        max_iterations = 10
        iteration = 0

        # Small models sometimes re-issue the identical action instead of
        # emitting a Final Answer. Track the last action to break such stalls.
        last_action_sig = None
        repeat_count = 0

        while iteration < max_iterations:
            iteration += 1
            user_prompt = build_bounded_prompt(
                history,
                f"\n\nIterative turn {iteration}/{max_iterations}. What is your next Action?",
            )

            try:
                # Get next action suggestion from LLM
                response = await self.llm_client.chat_completion(
                    system_prompt=BUILDER_SYSTEM_PROMPT, user_prompt=user_prompt
                )

                # Append model's response to history
                history.append(response)

                # Check if the model returned its final answer
                if "Final Answer:" in response:
                    final_part = response.split("Final Answer:", 1)[1].strip()
                    # Strip any accidental markdown wrap
                    if final_part.startswith("```"):
                        lines = final_part.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines[-1].startswith("```"):
                            lines = lines[:-1]
                        final_part = "\n".join(lines).strip()

                    parsed_answer = json.loads(final_part)
                    changes = parsed_answer.get("changes", [])

                    # Compute SHA-256 for each change reported
                    for change in changes:
                        file_path = change["file_path"]
                        full_path = self.tools._resolve_path(file_path)
                        if os.path.exists(full_path):
                            with open(full_path, "rb") as f:
                                sha256_hash = hashlib.sha256(f.read()).hexdigest()
                            change["sha256"] = sha256_hash
                        else:
                            change["sha256"] = ""

                    print(
                        f"[Builder] Task {task_id} complete. Reporting {len(changes)} change(s)."
                    )
                    await self.send_event(
                        event_type="TASK_COMPLETED",
                        recipient="orchestrator",
                        payload={"task_id": task_id, "changes": changes},
                    )
                    return

                # Parse tool action
                action_match = re.search(r"Action:\s*(\w+)\((.*)\)", response)
                if not action_match:
                    raise ValueError(
                        "LLM response did not output a valid Action block format."
                    )

                tool_name = action_match.group(1)
                tool_args_str = action_match.group(2)

                # Parse arguments dynamically as python dict format
                # Using a safe json/eval parse for named parameters like path="src/x.py"
                # A simple regex tokenizer is safer than raw eval
                args_dict = {}
                for arg_pair in re.finditer(
                    r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+))', tool_args_str
                ):
                    key = arg_pair.group(1)
                    val = (
                        arg_pair.group(2) or arg_pair.group(3) or int(arg_pair.group(4))
                    )
                    args_dict[key] = val

                # Detect a stalled loop: identical action repeated back-to-back.
                action_sig = f"{tool_name}:{json.dumps(args_dict, sort_keys=True)}"
                if action_sig == last_action_sig:
                    repeat_count += 1
                else:
                    repeat_count = 0
                    last_action_sig = action_sig

                if repeat_count >= 2:
                    print(
                        f"[Builder] Detected {repeat_count + 1}x repeated action; nudging toward Final Answer."
                    )
                    history.append(
                        "Observation: You have repeated the SAME action several times with no new "
                        "information. Do NOT repeat it. If your change is already written and verified, "
                        "output your Final Answer JSON now. Otherwise take a DIFFERENT action."
                    )
                    continue

                # Execute the matched tool
                print(f"[Builder] Action -> Calling {tool_name}({args_dict})")
                if tool_name == "read_file":
                    result = self.tools.read_file(args_dict.get("path", ""))
                elif tool_name == "write_file":
                    result = self.tools.write_file(
                        args_dict.get("path", ""), args_dict.get("content", "")
                    )
                elif tool_name == "grep_search":
                    result = self.tools.grep_search(
                        args_dict.get("pattern", ""), args_dict.get("directory", ".")
                    )
                elif tool_name == "execute_command":
                    result = self.tools.execute_command(args_dict.get("command", ""))
                else:
                    result = {
                        "status": "error",
                        "message": f"Unknown tool: '{tool_name}'",
                    }

                obs_str = truncate_observation(f"Observation: {json.dumps(result)}")
                print(f"[Builder] Observation outcome status: {result.get('status')}")
                history.append(obs_str)

            except Exception as e:
                err_msg = f"[Builder ERROR] Loop iteration {iteration} failed: {e}"
                print(err_msg)
                history.append(
                    f"Observation: Error executing tool or parsing response. Details: {e}"
                )

        # If loop limits are hit without finishing
        print(f"[Builder] Failed to complete task {task_id}: Iteration limit reached.")
        await self.send_event(
            event_type="TASK_FAILED",
            recipient="orchestrator",
            payload={
                "task_id": task_id,
                "reason": "Iteration budget limit exceeded without final answer.",
            },
        )
