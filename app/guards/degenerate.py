"""Degenerate-output guards: periodic repetition loops and CJK code-switching
inside code blocks/tool calls. Both are frontier-model failure modes seen in
production; detecting them lets the proxy end the stream cleanly instead of
forwarding garbage or splicing a retried answer onto a partial one."""

from typing import Optional

from app.config import _REP_MAX_UNIT, _REP_MIN_REPEATS, _REP_MIN_RUN, _CJK_CODE_MIN
from app.guards.cjk import cjk_in_code_blocks, cjk_in_tool_calls, prompt_has_cjk


def looping_tail(text: str) -> bool:
    """Full scan over the last 4000 chars for a short unit repeating >= _REP_MIN_REPEATS
    times contiguously, spanning at least _REP_MIN_RUN chars. Requires the unit to
    contain an alphanumeric so pure whitespace/punctuation runs don't false-positive."""
    if not text:
        return False
    tail = text[-4000:]
    n = len(tail)
    for unit_len in range(1, _REP_MAX_UNIT + 1):
        if unit_len * _REP_MIN_REPEATS > n:
            break
        unit = tail[-unit_len:]
        if not any(c.isalnum() for c in unit):
            continue
        repeats = 1
        pos = n - unit_len
        while pos - unit_len >= 0 and tail[pos - unit_len:pos] == unit:
            repeats += 1
            pos -= unit_len
        if repeats >= _REP_MIN_REPEATS and repeats * unit_len >= _REP_MIN_RUN:
            return True
    return False


def looping_suffix(text: str) -> bool:
    """Cheaper streaming variant of looping_tail: only checks the last 2000 chars."""
    if not text:
        return False
    return looping_tail(text[-2000:])


def degenerate_reason(
    text: str,
    finish_reason: Optional[str],
    body: dict,
    msg: Optional[dict] = None,
) -> Optional[str]:
    if finish_reason == "repetition":
        return "repetition"
    if looping_tail(text):
        return "repetition"
    if prompt_has_cjk(body.get("messages") or []):
        return None
    if cjk_in_code_blocks(text, _CJK_CODE_MIN):
        return "cjk_in_code"
    if msg and cjk_in_tool_calls(msg.get("tool_calls"), _CJK_CODE_MIN):
        return "cjk_in_code"
    return None
