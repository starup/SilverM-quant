"""
Anthropic Claude LLM适配器 - Claude模型适配器实现
"""
import requests
from typing import ClassVar, Dict, List, Optional
from agent_integration.llm_adapters.base import OpenAICompatibleBase


class ChatClaude(OpenAICompatibleBase):
    provider_name: ClassVar[str] = "anthropic"
    default_model: ClassVar[str] = "claude-sonnet-4-20250514"
    api_base: ClassVar[str] = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str = None, model: str = None, **kwargs):
        import os
        base_url = kwargs.pop('base_url', None) or os.environ.get("ANTHROPIC_BASE_URL", "")
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or None,
            **kwargs
        )
        self._last_usage = None

    def _get_api_key_from_env(self) -> str:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def _llm_type(self) -> str:
        return "chat_anthropic"

    def _generate_content(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                api_messages.append(msg)

        if not api_messages:
            api_messages = [{"role": "user", "content": "Hello"}]

        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
        }

        if system_text.strip():
            payload["system"] = system_text.strip()
        if stop:
            payload["stop_sequences"] = stop
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        self._last_usage = data.get("usage")

        content_blocks = data.get("content", [])
        text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
        return "\n".join(text_parts)

    def _parse_usage_from_response(self, response_text: str, **kwargs):
        if self._last_usage:
            return {
                'prompt_tokens': self._last_usage.get('input_tokens', 0),
                'completion_tokens': self._last_usage.get('output_tokens', 0)
            }
        return None
