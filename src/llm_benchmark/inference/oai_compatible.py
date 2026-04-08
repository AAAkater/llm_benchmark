"""OpenAI Batch API client for bulk inference.

This module provides utilities for using OpenAI's Batch API to process
large numbers of requests efficiently.
"""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

from llm_benchmark.utils import logger


class InferenceResult(BaseModel):
    """Result of a single inference request."""

    response: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0  # Time in milliseconds (from first chunk to end)
    ttft_ms: float = 0.0  # Time to first token in milliseconds

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self.input_tokens + self.output_tokens

    @property
    def output_tps(self) -> float:
        """Output tokens per second (calculated from first chunk)."""
        if self.latency_ms > 0 and self.output_tokens > 0:
            return self.output_tokens / (self.latency_ms / 1000.0)
        return 0.0

    @property
    def total_tps(self) -> float:
        """Total tokens per second (calculated from first chunk)."""
        if self.latency_ms > 0 and self.total_tokens > 0:
            return self.total_tokens / (self.latency_ms / 1000.0)
        return 0.0


class OAIBatchClient:
    """Client for OpenAI Batch API operations.

    Accepts an AsyncOpenAI instance directly, allowing users to configure
    API connection parameters (base_url, api_key, timeout) themselves.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        enable_thinking: bool = False,
    ):
        self._client = client
        self.model_name = model_name
        self.enable_thinking = enable_thinking

    async def query_single(
        self,
        prompt: str,
    ) -> InferenceResult:
        """Query the model with a single prompt using streaming.

        Args:
            prompt: The prompt to send.

        Returns:
            InferenceResult with response, token counts, and latency.
        """
        import time

        logger.debug(f"Sending request with prompt length: {len(prompt)}")
        request_start_time = time.perf_counter()

        extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if self.enable_thinking else None

        # 流式输出（不打印，只收集）
        stream_resp = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},  # Request token usage in stream
            extra_body=extra_body,
        )
        chunks = []
        input_tokens = 0
        output_tokens = 0
        first_chunk_time = None
        async for chunk in stream_resp:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
            # Get usage from the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
        response = "".join(chunks).strip()

        # 计算 TTFT (Time To First Token) 和生成延迟
        if first_chunk_time is not None:
            ttft_ms = (first_chunk_time - request_start_time) * 1000
            latency_ms = (time.perf_counter() - first_chunk_time) * 1000
        else:
            ttft_ms = 0.0
            latency_ms = 0.0

        return InferenceResult(
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
        )

    async def query_batch_concurrent(
        self,
        prompts: list[str],
        max_concurrent: int = 10,
    ) -> list[InferenceResult]:
        """Query multiple prompts concurrently (non-batch API).

        This uses regular async requests instead of the Batch API,
        useful for smaller batches or when Batch API is not available.

        Args:
            prompts: List of prompts.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of InferenceResult in order.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def query_with_semaphore(prompt: str) -> InferenceResult:
            async with semaphore:
                try:
                    return await self.query_single(prompt)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    return InferenceResult(response=f"ERROR: {e}")

        tasks = [query_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)
