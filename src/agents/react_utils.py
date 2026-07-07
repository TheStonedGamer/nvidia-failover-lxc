"""Helpers for keeping a ReAct agent's scratchpad inside the model context window.

The Builder and Reviewer agents run a Thought/Action/Observation loop and append
every model response and tool observation to a growing `history` list. Tool
observations can carry whole file contents, so an unbounded history quickly blows
past the backend context size (observed: 14k+ tokens on a 32k model after only a
few `read_file`/`write_file` turns). These helpers cap each observation and keep
only a rolling window of recent turns while always preserving the task framing.
"""

from typing import List

# Max characters kept from a single tool observation before it is elided.
MAX_OBS_CHARS = 4000
# Number of leading history entries always kept (task description + diagnostics).
HISTORY_HEAD = 2
# Number of most-recent history entries kept beyond the head.
HISTORY_TAIL = 8


def truncate_observation(text: str, limit: int = MAX_OBS_CHARS) -> str:
    """Clip an oversized observation, keeping head and tail for context."""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    elided = len(text) - limit
    return f"{head}\n...[{elided} chars elided to fit context]...\n{tail}"


def build_bounded_prompt(
    history: List[str],
    suffix: str,
    head: int = HISTORY_HEAD,
    tail: int = HISTORY_TAIL,
) -> str:
    """Render `history` into a prompt, keeping the first `head` and last `tail` entries.

    When the middle is dropped, a marker records how many turns were elided so the
    model knows its scratchpad was trimmed rather than reset.
    """
    if len(history) <= head + tail:
        kept = history
    else:
        dropped = len(history) - head - tail
        kept = (
            history[:head]
            + [f"...[{dropped} earlier turn(s) elided to fit context]..."]
            + history[-tail:]
        )
    return "\n\n".join(kept) + suffix
