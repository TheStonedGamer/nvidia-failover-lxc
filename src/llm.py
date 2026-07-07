import json
import urllib.request
import urllib.error
import asyncio
from typing import Optional

class LocalLLMClient:
    """Universal client for local LLM engines supporting OpenAI-compatible APIs (Ollama, Llama.cpp, vLLM)."""
    def __init__(self, base_url: str = "http://localhost:8080/v1", default_model: str = "qwen3-4b"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def chat_completion(self, system_prompt: str, user_prompt: str, model: Optional[str] = None, json_mode: bool = False) -> str:
        """Asynchronously requests a chat completion from the local inference server."""
        return await asyncio.to_thread(self._chat_completion_sync, system_prompt, user_prompt, model or self.default_model, json_mode)

    def _chat_completion_sync(self, system_prompt: str, user_prompt: str, model: str, json_mode: bool) -> str:
        url = f"{self.base_url}/chat/completions"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False
        }
        
        # Enable structured JSON mode if supported
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                return res_json["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"LLM Server returned error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to local inference engine at {url}.\n"
                f"Ensure Ollama, Llama.cpp, or vLLM is running. Error details: {e}"
            )
