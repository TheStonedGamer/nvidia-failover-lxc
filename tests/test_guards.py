from app.guards.degenerate import looping_tail, looping_suffix, degenerate_reason
from app.guards.cjk import (
    cjk_count,
    prompt_has_cjk,
    cjk_in_code_blocks,
    cjk_in_tool_calls,
)


def test_looping_tail_detects_short_repeat():
    assert looping_tail("ha" * 50) is True


def test_looping_tail_ignores_normal_text():
    assert looping_tail("The quick brown fox jumps over the lazy dog.") is False


def test_looping_tail_empty_text():
    assert looping_tail("") is False


def test_looping_tail_requires_alnum_unit():
    # Pure punctuation/whitespace repeats should never count as looping.
    assert looping_tail(" . " * 100) is False


def test_looping_suffix_only_checks_last_2000_chars():
    # A repeat buried far before the tail window should not be flagged; the
    # padding itself must not contain a repeat of its own.
    import random

    rng = random.Random(0)
    padding = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(3000))
    padded = ("ha" * 50) + padding
    assert looping_suffix(padded) is False


def test_cjk_count_basic():
    assert cjk_count("hello") == 0
    assert cjk_count("你好") == 2


def test_prompt_has_cjk_checks_user_and_system_only():
    messages = [
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "hello"},
    ]
    assert prompt_has_cjk(messages) is False
    messages[1]["content"] = "你好"
    assert prompt_has_cjk(messages) is True


def test_cjk_in_code_blocks_only_flags_inside_fences():
    text = "prose 你好你好 more prose ```code 你好你好``` trailing"
    assert cjk_in_code_blocks(text, 2) is True
    assert cjk_in_code_blocks("你好你好 no fences here", 2) is False


def test_cjk_in_tool_calls():
    tool_calls = [{"function": {"arguments": '{"x": "你好你好"}'}}]
    assert cjk_in_tool_calls(tool_calls, 2) is True
    assert cjk_in_tool_calls(None, 2) is False


def test_degenerate_reason_repetition_flag():
    assert degenerate_reason("", "repetition", {}) == "repetition"


def test_degenerate_reason_looping_text():
    assert degenerate_reason("ha" * 50, None, {}) == "repetition"


def test_degenerate_reason_cjk_in_code_when_prompt_has_none():
    text = "```py\n你好你好\n```"
    body = {"messages": [{"role": "user", "content": "write some code"}]}
    assert degenerate_reason(text, None, body) == "cjk_in_code"


def test_degenerate_reason_skips_cjk_when_prompt_has_cjk():
    text = "```py\n你好你好\n```"
    body = {"messages": [{"role": "user", "content": "你好, 写代码"}]}
    assert degenerate_reason(text, None, body) is None


def test_degenerate_reason_clean_output():
    body = {"messages": [{"role": "user", "content": "hello"}]}
    assert degenerate_reason("a normal helpful answer", "stop", body) is None
