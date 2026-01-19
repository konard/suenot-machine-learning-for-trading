"""
LLM Client for Prompt Engineering Trading

Provides a unified interface for interacting with various LLM APIs
including OpenAI, Anthropic, and local models via Ollama.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import asyncio

try:
    import httpx
except ImportError:
    httpx = None

try:
    import aiohttp
except ImportError:
    aiohttp = None


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate completion for the given prompt."""
        pass

    @abstractmethod
    async def complete_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate completion with full metadata."""
        pass


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client.

    Supports GPT-4 and GPT-3.5-turbo models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate completion."""
        response = await self.complete_with_metadata(
            prompt, temperature, max_tokens, **kwargs
        )
        return response.content

    async def complete_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate completion with metadata."""
        if httpx is None:
            raise ImportError("httpx is required for OpenAI client")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()

        choice = result["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=result["model"],
            usage=result.get("usage"),
            finish_reason=choice.get("finish_reason")
        )


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client.

    Supports Claude models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        base_url: str = "https://api.anthropic.com/v1"
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate completion."""
        response = await self.complete_with_metadata(
            prompt, temperature, max_tokens, **kwargs
        )
        return response.content

    async def complete_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate completion with metadata."""
        if httpx is None:
            raise ImportError("httpx is required for Anthropic client")

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()

        return LLMResponse(
            content=result["content"][0]["text"],
            model=result["model"],
            usage=result.get("usage"),
            finish_reason=result.get("stop_reason")
        )


class OllamaClient(BaseLLMClient):
    """
    Ollama client for local LLM models.

    Supports any model available in Ollama.
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate completion."""
        response = await self.complete_with_metadata(
            prompt, temperature, max_tokens, **kwargs
        )
        return response.content

    async def complete_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate completion with metadata."""
        if httpx is None:
            raise ImportError("httpx is required for Ollama client")

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=120.0
            )
            response.raise_for_status()
            result = response.json()

        return LLMResponse(
            content=result["response"],
            model=result["model"],
            usage=None,
            finish_reason="stop" if result.get("done") else None
        )


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing and demos.

    Returns predefined responses based on prompt content.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.default_response = json.dumps({
            "sentiment": "NEUTRAL",
            "confidence": 50,
            "signal": "HOLD",
            "reasoning": "Mock response for testing"
        })

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate mock completion."""
        # Check for matching response
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        return self.default_response

    async def complete_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate mock completion with metadata."""
        content = await self.complete(prompt, temperature, max_tokens, **kwargs)
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())},
            finish_reason="stop"
        )


def create_llm_client(
    provider: str = "mock",
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to create LLM client.

    Args:
        provider: LLM provider ("openai", "anthropic", "ollama", "mock")
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM client
    """
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
        "mock": MockLLMClient
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")

    return providers[provider](**kwargs)
