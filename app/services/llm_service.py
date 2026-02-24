"""
LLM service for text generation using Ollama.

Provides an abstraction layer over the Ollama API
with proper error handling and configuration.
"""

from typing import AsyncGenerator, Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    Ollama-based LLM service for text generation.

    Supports both synchronous and streaming generation.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]

        except httpx.ConnectError:
            raise LLMServiceError(
                message="Cannot connect to Ollama. Is it running?",
                details={
                    "base_url": self.base_url,
                    "hint": "Run 'ollama serve' or check if Ollama is installed",
                },
            )
        except httpx.TimeoutException:
            raise LLMServiceError(
                message="LLM request timed out",
                details={"timeout": self.timeout},
            )
        except httpx.HTTPStatusError as e:
            raise LLMServiceError(
                message=f"Ollama API error: {e.response.status_code}",
                details={"response": e.response.text},
            )
        except Exception as e:
            raise LLMServiceError(
                message=f"LLM generation failed: {str(e)}",
            )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text completion with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Generated text chunks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    yield content

        except httpx.ConnectError:
            raise LLMServiceError(
                message="Cannot connect to Ollama. Is it running?",
                details={"base_url": self.base_url},
            )
        except Exception as e:
            raise LLMServiceError(
                message=f"LLM streaming failed: {str(e)}",
            )

    async def health_check(self) -> bool:
        """Check if Ollama service is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            raise LLMServiceError(
                message=f"Failed to list models: {str(e)}",
            )
