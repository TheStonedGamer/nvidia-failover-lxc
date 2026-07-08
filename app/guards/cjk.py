"""CJK-in-code detection: catches models that code-switch to Chinese inside
generated source or tool-call arguments (a known frontier-model failure mode).
Only fires when the user's own prompt had no CJK, to avoid false positives
when the user is legitimately writing in Chinese/Japanese/Korean."""

import re
from typing import List

_CJK_RE = re.compile(
    "[぀-ヿ㐀-䶿一-鿿㐀-䶿豈-﫿ｦ-ﾟ가-힯]"
)


def cjk_count(text: str) -> int:
    if not text:
        return 0
    return len(_CJK_RE.findall(text))


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def prompt_has_cjk(messages: List[dict]) -> bool:
    for m in messages or []:
        if m.get("role") not in ("user", "system"):
            continue
        if cjk_count(_content_to_text(m.get("content"))) > 0:
            return True
    return False


def cjk_in_code_blocks(text: str, min_count: int) -> bool:
    if not text:
        return False
    parts = text.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1 and cjk_count(part) >= min_count:
            return True
    return False


def cjk_in_tool_calls(tool_calls, min_count: int) -> bool:
    if not tool_calls:
        return False
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str) and cjk_count(args) >= min_count:
            return True
    return False
