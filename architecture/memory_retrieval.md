# Memory Architecture & Retrieval Engine

This specification details how knowledge is stored, indexed, retrieved, and cached across the platform.

---

## Storage Schemas

### 1. Working Memory (SQLite Database)
Working memory captures active developer sessions, tool executions, and file state histories.

```sql
CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT -- 'active', 'suspended', 'completed'
);

CREATE TABLE active_task_dag (
    task_id TEXT PRIMARY KEY,
    session_id TEXT,
    description TEXT,
    status TEXT, -- 'todo', 'in_progress', 'completed', 'failed'
    verifiable_condition TEXT,
    dependencies TEXT, -- JSON array of task_ids
    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
);

CREATE TABLE workspace_changes (
    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    file_path TEXT,
    original_sha256 TEXT,
    modified_sha256 TEXT,
    diff_content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
);
```

### 2. Project Memory (Obsidian Markdown Vault)
Stored in an Obsidian vault at `.obsidian/project-memory/` inside the project. Uses double brackets `[[WikiLinks]]` for cross-references.

```
project-memory/
├── 00_meta/
│   ├── index.md             # Entrance point, links to modules, specs, conventions
│   └── project_meta.md       # High-level stack, entry points, build/deploy commands
├── 01_architecture/
│   ├── auth_flow.md         # Design doc detailing how authentication behaves
│   └── database_schema.md   # Data models, migrations, relationships
├── 02_apis/
│   ├── external_services.md # External APIs, endpoints, authentication keys
│   └── internal_contracts.md# Internal service contracts and RPC payloads
└── 03_conventions/
    ├── coding_style.md      # Language-specific style rules, lint configs
    └── review_rules.md      # Testing policies, CI requirements
```

*Example Index Link:* `Check [[auth_flow]] and [[internal_contracts]] for integration rules.`

### 3. Long-Term Memory (LanceDB / Vector DB)
Stores semantic representations of historical code fixes, user preferences, and solutions.
*   **Vector Database Schema**:
    ```json
    {
      "id": "uuid",
      "embedding": [0.154, -0.923, 0.021, "... 384 dimensions"],
      "metadata": {
        "problem_description": "Connection leak in Postgres client when handling query timeouts",
        "solution_rationale": "Wrap client queries in a try-finally block to ensure connection release",
        "affected_files": ["src/db/client.py"],
        "diff": "@@ -12,4 +12,8 @@ ...",
        "timestamp": "2026-07-05T15:20:00"
      }
    }
    ```

---

## Retrieval & Search Execution

To feed the Context OS, the search engine runs a hybrid retrieval pipeline:

```python
def retrieve_knowledge(query, session_id):
    # 1. Semantic Vector Query (using local BGE-M3 model)
    query_vector = embed_model.embed(query)
    semantic_results = vector_db.search(query_vector).limit(10)
    
    # 2. Keyword Search (using SQLite FTS5 index of project files)
    keyword_results = sqlite_fts.search(query).limit(10)
    
    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    for rank, doc in enumerate(semantic_results):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + (1.0 / (rank + 60))
        
    for rank, doc in enumerate(keyword_results):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + (1.0 / (rank + 60))
        
    # Sort documents by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Reranking (Using a local cross-encoder model)
    top_candidates = [get_doc_by_id(doc_id) for doc_id, _ in sorted_docs[:8]]
    reranked_results = cross_encoder.rerank(query, top_candidates)
    
    return reranked_results
```

---

## Semantic Cache System

To prevent repetitive execution of complex reasoning models for questions on identical code components, we cache agent reasoning outputs.

```sql
CREATE TABLE semantic_cache (
    cache_key TEXT PRIMARY KEY,       -- Normalized hash of query + scope AST signature
    raw_query TEXT,
    cached_response TEXT,
    token_cost INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cache_dependency_files (
    cache_key TEXT,
    file_path TEXT,
    file_sha256 TEXT,
    PRIMARY KEY(cache_key, file_path),
    FOREIGN KEY(cache_key) REFERENCES semantic_cache(cache_key) ON DELETE CASCADE
);
```

### Cache Invalidation Protocol
1.  On file change event for `file_path`:
    *   Compute new SHA-256 of `file_path`.
    *   Compare with `file_sha256` in `cache_dependency_files`.
    *   If they differ, execute:
        ```sql
        DELETE FROM semantic_cache WHERE cache_key IN (
            SELECT cache_key FROM cache_dependency_files WHERE file_path = :file_path AND file_sha256 != :new_sha
        );
        ```
    *   This automatically cascades and cleans up stale reasoning blocks.
