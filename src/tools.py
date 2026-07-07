import subprocess
import os
import re
from typing import Dict, Any

class WorkspaceTools:
    """Core tools provided to the Builder and Reviewer agents to modify and verify the workspace."""
    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)

    def _resolve_path(self, path: str) -> str:
        """Helper to ensure paths are absolute and resolved within the workspace root."""
        abs_path = os.path.abspath(os.path.join(self.workspace_root, path))
        if not abs_path.startswith(self.workspace_root):
            raise PermissionError(f"Access denied: path '{path}' is outside the workspace root.")
        return abs_path

    def read_file(self, path: str) -> Dict[str, Any]:
        """Reads the entire content of a file in the workspace."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return {"status": "error", "message": f"File '{path}' does not exist."}
            
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "content": content}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Creates or overwrites a file with the specified content."""
        try:
            full_path = self._resolve_path(path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "message": f"File '{path}' written successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def grep_search(self, pattern: str, directory: str = ".") -> Dict[str, Any]:
        """Searches files in the directory for a regular expression pattern."""
        try:
            full_dir = self._resolve_path(directory)
            results = []
            regex = re.compile(pattern)
            
            for root, _, files in os.walk(full_dir):
                for file in files:
                    # Skip typical directories (git, cache, venv)
                    if any(ignored in root for ignored in [".git", "__pycache__", ".venv", "node_modules"]):
                        continue
                        
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.workspace_root)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    results.append({
                                        "file": rel_path,
                                        "line_number": i,
                                        "content": line.strip()
                                    })
                    except Exception:
                        continue
                        
            return {"status": "success", "matches": results[:100]}  # Cap at 100 matches
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_command(self, command: str, timeout_seconds: int = 30) -> Dict[str, Any]:
        """Executes a terminal command within the workspace root directory."""
        try:
            # Execute command inside powershell or cmd on Windows, sh on POSIX
            is_windows = os.name == "nt"
            shell = "powershell.exe" if is_windows else "/bin/sh"
            
            # Run subprocess with timeout
            process = subprocess.run(
                [shell, "-Command", command] if is_windows else [shell, "-c", command],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            return {
                "status": "success",
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Command timed out after {timeout_seconds} seconds."}
        except Exception as e:
            return {"status": "error", "message": str(e)}
