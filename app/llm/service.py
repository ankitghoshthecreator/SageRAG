"""
Multi-Provider LLM Service
===========================
Supports: openai | gemini | groq | openrouter | local (echo stub)

Usage
-----
    from app.llm.service import llm_service, LLMMessage

    messages = [LLMMessage(role="user", content="Hello!")]
    answer   = llm_service.generate(messages)

    # Streaming (returns an async generator of token strings)
    async for token in llm_service.stream(messages):
        print(token, end="", flush=True)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

from app.utils.config import settings

logger = logging.getLogger("sagerag.llm")


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class LLMMessage:
    role: str       # "system" | "user" | "assistant"
    content: str


# ── Provider implementations ────────────────────────────────────────────────────

class _OpenAIProvider:
    def __init__(self):
        import openai
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"OpenAI provider ready (model={settings.LLM_MODEL_NAME})")

    def _to_dict(self, msgs: List[LLMMessage]):
        return [{"role": m.role, "content": m.content} for m in msgs]

    def generate(self, messages: List[LLMMessage], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=self._to_dict(messages),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=self._to_dict(messages),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class _GeminiProvider:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
        logger.info(f"Gemini provider ready (model={settings.LLM_MODEL_NAME})")

    def _to_gemini(self, messages: List[LLMMessage]):
        """Convert messages to Gemini format, building a chat history."""
        history = []
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't have a system role; prepend to first user message
                prompt = msg.content + "\n\n"
            elif msg.role == "user":
                history.append({"role": "user", "parts": [prompt + msg.content]})
                prompt = ""
            elif msg.role == "assistant":
                history.append({"role": "model", "parts": [msg.content]})
        return history

    def generate(self, messages: List[LLMMessage], **kwargs) -> str:
        history = self._to_gemini(messages)
        if not history:
            return ""
        # Last message is the user turn
        last_user = history[-1]["parts"][0] if history[-1]["role"] == "user" else ""
        chat = self._model.start_chat(history=history[:-1])
        response = chat.send_message(last_user)
        return response.text or ""

    async def stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        history = self._to_gemini(messages)
        last_user = history[-1]["parts"][0] if history[-1]["role"] == "user" else ""
        chat = self._model.start_chat(history=history[:-1])
        response = chat.send_message(last_user, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


class _GroqProvider:
    def __init__(self):
        from groq import Groq
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info(f"Groq provider ready (model={settings.LLM_MODEL_NAME})")

    def _to_dict(self, msgs: List[LLMMessage]):
        return [{"role": m.role, "content": m.content} for m in msgs]

    def generate(self, messages: List[LLMMessage], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=self._to_dict(messages),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=self._to_dict(messages),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class _OpenRouterProvider:
    """Calls OpenRouter via the OpenAI-compatible API."""
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        import openai
        self._client = openai.OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=self.BASE_URL,
        )
        logger.info(f"OpenRouter provider ready (model={settings.LLM_MODEL_NAME})")

    def _to_dict(self, msgs: List[LLMMessage]):
        return [{"role": m.role, "content": m.content} for m in msgs]

    def generate(self, messages: List[LLMMessage], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=self._to_dict(messages),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=self._to_dict(messages),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class _LocalEchoProvider:
    """Development stub — echoes back the last user message."""
    def __init__(self):
        logger.warning("Using local echo LLM provider (no real generation). Set LLM_PROVIDER in .env")

    def generate(self, messages: List[LLMMessage], **kwargs) -> str:
        user_msgs = [m.content for m in messages if m.role == "user"]
        return f"[LocalEcho] You asked: {user_msgs[-1] if user_msgs else '(no message)'}"

    async def stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        text = self.generate(messages)
        for word in text.split(" "):
            yield word + " "


# ── LLM Service facade ─────────────────────────────────────────────────────────

class LLMService:
    """
    Provider-agnostic facade.
    The underlying provider is created lazily on first call.
    """

    def __init__(self):
        self._provider = None

    def _get_provider(self):
        if self._provider is not None:
            return self._provider

        provider_name = settings.LLM_PROVIDER.lower()
        try:
            if provider_name == "openai":
                self._provider = _OpenAIProvider()
            elif provider_name == "gemini":
                self._provider = _GeminiProvider()
            elif provider_name == "groq":
                self._provider = _GroqProvider()
            elif provider_name == "openrouter":
                self._provider = _OpenRouterProvider()
            else:
                self._provider = _LocalEchoProvider()
        except Exception as e:
            logger.error(f"Failed to init LLM provider '{provider_name}': {e}. Falling back to echo.")
            self._provider = _LocalEchoProvider()

        return self._provider

    def generate(self, messages: List[LLMMessage], **kwargs) -> str:
        """Synchronous generation. Returns the full response string."""
        return self._get_provider().generate(messages, **kwargs)

    async def stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        """Async generator that yields token strings."""
        async for token in self._get_provider().stream(messages, **kwargs):
            yield token


# Module-level singleton
llm_service = LLMService()
