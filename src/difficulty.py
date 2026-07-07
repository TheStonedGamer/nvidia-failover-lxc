"""Free static difficulty heuristic (Phase 2.8).

NOT a router — the failure trigger decides *whether* to escalate. This hint
only tunes (1) how many local attempts a task gets before escalating and
(2) which external tier to try first once escalating. No model calls; only
signals we already have: task text keywords and the planner DAG's shape.
"""

import re
from typing import Optional

_HARD = re.compile(
    r"concurren|race condition|deadlock|migrat|refactor across|multi-?file|"
    r"crypto|security audit|memory leak|performance|distributed|protocol|"
    r"debug|undefined behavior|架构|architecture",
    re.IGNORECASE,
)
_EASY = re.compile(
    r"rename|typo|add (a )?field|comment|docstring|format|bump version|"
    r"write (a )?test for|single function|one-?liner",
    re.IGNORECASE,
)


def score(description: str, dag: Optional[dict] = None) -> float:
    """Difficulty in [0, 1] from static signals. 0.5 = unknown/neutral."""
    s = 0.5
    hard_hits = len(_HARD.findall(description))
    easy_hits = len(_EASY.findall(description))
    s += 0.15 * hard_hits - 0.15 * easy_hits

    if dag:
        tasks = dag.get("tasks", []) or []
        n = len(tasks)
        # Wide/deep plans are harder. Depth ~= longest dependency chain.
        s += min(0.2, 0.04 * max(0, n - 3))
        deps = {t.get("id"): t.get("depends_on", []) or [] for t in tasks}

        def depth(tid, seen=frozenset()):
            if tid in seen:
                return 0  # cycle guard
            return 1 + max(
                (depth(d, seen | {tid}) for d in deps.get(tid, [])), default=0
            )

        max_depth = max((depth(t) for t in deps), default=0)
        s += min(0.15, 0.05 * max(0, max_depth - 2))

    return max(0.0, min(1.0, s))


def local_attempts(difficulty: float, base: int = 3) -> int:
    """Retry budget: easy tasks get fewer local retries before escalating;
    hard-flagged tasks escalate after 1 failure instead of grinding."""
    if difficulty >= 0.75:
        return 1
    if difficulty <= 0.3:
        return max(1, base - 1)
    return base
