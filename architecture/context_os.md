# Context OS

Context OS is the out-of-model orchestration layer that structures and optimizes prompts. It ensures that the model is only fed highly relevant information, minimizing context waste and preventing hallucinations.

---

## Architectural Workflow

```
[Raw Agent Call] 
       │
       ▼
 1. Resolve State ────────► Fetch Active Task, AST, and Diagnostics
       │
       ▼
 2. Retrieve Assets ──────► Run Semantic, Keyword & Import Dependency Search
       │
       ▼
 3. Score & Rerank ───────► Drop Low-scoring chunks to meet Token Budget
       │
       ▼
 4. Compile AST ──────────► Convert large files to Symbol Signature outlines
       │
       ▼
 5. Output Prompt ────────► Render formatted Markdown payload to Model
```

---

## Language Server Protocol (LSP) Integration

Context OS communicates with local language servers (e.g. `pyright`, `tsserver`, `gopls`, `rust-analyzer`) via JSON-RPC over stdin/stdout.

### Use Cases
*   **Compile Diagnostics**: Automatically pull warning/error messages at the target file location, injecting them directly into the Builder's context block when debugging.
*   **Jump-to-Definition**: Follow symbols imported in the modified files to extract the parent definition signatures without loading the entire dependency files.
*   **Symbol Outlines**: Query LSP document symbols to locate where classes/methods start and end, allowing precise chunking for the vector indexing engine.

---

## AST Parser (Tree-Sitter)

Context OS uses Tree-sitter to parse source files into Abstract Syntax Trees (AST). This allows structural modifications of code context before model insertion.

### Smart Outline Extraction
Instead of passing a 1,000-line implementation file, Context OS generates a skeleton outline of unmodified helper files:

```python
# AST Outline representation for helper.py (Auto-extracted by Context OS)
class DatabaseClient:
    def __init__(self, connection_string: str): ...
    def execute_query(self, query: str, params: dict = None) -> list: ...
    def close(self) -> None: ...
```

This reduces the token usage of supporting files by up to 90%, leaving the full details only for files marked for modification.

---

## The Prompt Compilation Pipeline

The prompt compiler runs the following steps to construct a prompt:

1.  **Extract Focal Points**: Identify target files from the user's request and the active Planner task.
2.  **Gather Workspace Diagnostics**:
    ```python
    diagnostics = lsp.get_diagnostics(file_path)
    # Filter for errors and warnings in active files
    active_errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    ```
3.  **Trace Code Imports**: Parse imports in focal files. For each import within the project root, retrieve its signature using the AST parser.
4.  **Enforce Adaptive Token Budget**:
    *   Retrieve the context limit of the model chosen by the Router (e.g., 8,192 tokens for `qwen-2.5-coder-7b`).
    *   Calculate and allocate token partitions using the budget algorithm.
    *   If remaining tokens are insufficient, slice older conversation history first. If still insufficient, replace lower-scoring vector retrieval chunks with their AST signatures.
5.  **Compile & Render**: Merges the system prompt, environment vars, current task, files, diagnostics, and historical memories into the final markdown payload.
