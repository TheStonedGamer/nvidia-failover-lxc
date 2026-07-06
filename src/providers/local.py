"""Local provider — wraps the existing LocalLLMClient (llama.cpp on :8080).

Always the first rung. Available whenever the llama.cpp server responds.
"""

import urllib.request
from typing import Optional

from src.llm import LocalLLMClient
from src.providers.base import Provider


class LocalProvider(Provider):
    name = "local"
    shape = "chat"

    def __init__(self, client: Optional[LocalLLMClient] = None):
        self.client = client or LocalLLMClient()

    def _probe(self) -> tuple:
        url = f"{self.client.base_url.rsplit('/v1', 1)[0]}/health"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True, "ok"
                return False, f"health returned {r.status}"
        except OSError as e:
            return False, f"unreachable: {e}"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        return await self.client.chat_completion(
            system_prompt, user_prompt, model=model, json_mode=json_mode
        )
