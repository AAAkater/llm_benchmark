"""OpenAI Batch API client for bulk inference.

This module provides utilities for using OpenAI's Batch API to process
large numbers of requests efficiently.
"""

import asyncio
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_benchmark.utils.logger import logger


class SamplingConfig(BaseModel):
    """Sampling parameters for LLM generation."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=256, gt=0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters dict."""
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        return params


class ClientConfig(BaseSettings):
    """Configuration for the OpenAI client."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="Base URL for the OpenAI-compatible API server",
    )
    api_key: str = Field(
        default="sk-dummy",
        description="API key for authentication",
    )
    model_name: str = Field(
        default="local-model",
        description="Name of the model to use",
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable thinking/reasoning mode for the model",
    )


class BatchClient:
    """Client for OpenAI Batch API operations."""

    def __init__(self, config: ClientConfig | None = None):
        self.config = config or ClientConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    async def query_single(
        self,
        prompt: str,
        sampling: SamplingConfig | None = None,
        stream: bool = True,
    ) -> str:
        """Query the model with a single prompt (non-batch).

        Args:
            prompt: The prompt to send.
            sampling: Sampling configuration.
            stream: Whether to use streaming output.

        Returns:
            Model response.
        """
        sampling = sampling or SamplingConfig()

        logger.debug(f"Sending request with prompt length: {len(prompt)}")

        extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if self.config.enable_thinking else None

        if stream:
            # 流式输出（不打印，只收集）
            stream_resp = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body=extra_body,
                **sampling.to_api_params(),
            )
            chunks = []
            async for chunk in stream_resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            return "".join(chunks).strip()
        else:
            # 非流式
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra_body,
                **sampling.to_api_params(),
            )
            logger.debug("Received response")
            content = response.choices[0].message.content
            return content.strip() if content else ""

    async def query_batch_concurrent(
        self,
        prompts: list[str],
        sampling: SamplingConfig | None = None,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Query multiple prompts concurrently (non-batch API).

        This uses regular async requests instead of the Batch API,
        useful for smaller batches or when Batch API is not available.

        Args:
            prompts: List of prompts.
            sampling: Sampling configuration.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of responses in order.
        """
        sampling = sampling or SamplingConfig()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def query_with_semaphore(prompt: str) -> str:
            async with semaphore:
                try:
                    return await self.query_single(prompt, sampling)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    return f"ERROR: {e}"

        tasks = [query_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)
