"""Groq Cloud API LLM adapter implementation (Fast & Free Tier)."""

import logging
from typing import Iterator, Optional
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.GroqLLM")


class GroqLLM(BaseLLM):
    """Generative LLM using Groq Cloud's ultra-fast free LPU inference API.
    
    Supports models like llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.
    """

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided to use GroqLLM")
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    @property
    def provider_name(self) -> str:
        return f"Groq ({self.model_name})"

    def _get_client(self):
        if self._client is None:
            import httpx
            from groq import Groq
            self._client = Groq(api_key=self.api_key, http_client=httpx.Client())
        return self._client

    def _clean_response(self, text: str) -> str:
        """Strip internal thinking/reasoning tags if present in model output."""
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 500),
        )
        raw_text = response.choices[0].message.content or ""
        return self._clean_response(raw_text)

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 500),
            stream=True
        )
        inside_think = False
        think_buffer = ""
        prefix_buffer = ""
        stream_started = False

        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue

            if inside_think:
                think_buffer += delta
                if "</think>" in think_buffer:
                    inside_think = False
                    after_think = think_buffer.split("</think>", 1)[1]
                    think_buffer = ""
                    after_think = after_think.lstrip("\r\n")
                    if after_think:
                        yield after_think
                continue

            if not stream_started:
                prefix_buffer += delta
                if prefix_buffer.startswith("<think>"):
                    inside_think = True
                    stream_started = True
                    think_buffer = prefix_buffer
                    prefix_buffer = ""
                    if "</think>" in think_buffer:
                        inside_think = False
                        after_think = think_buffer.split("</think>", 1)[1]
                        think_buffer = ""
                        after_think = after_think.lstrip("\r\n")
                        if after_think:
                            yield after_think
                    continue
                if "<think>".startswith(prefix_buffer):
                    continue
                stream_started = True
                yield prefix_buffer
                prefix_buffer = ""
                continue

            yield delta

        if prefix_buffer and not inside_think:
            yield prefix_buffer
