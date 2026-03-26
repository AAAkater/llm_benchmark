"""OpenAI Batch API client for bulk inference.

This module provides utilities for using OpenAI's Batch API to process
large numbers of requests efficiently.
"""

import asyncio
import json
import time
from pathlib import Path
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
    stop: list[str] | None = Field(default=None)

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters dict."""
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.stop:
            params["stop"] = self.stop
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


class BatchRequest(BaseModel):
    """A single request in a batch."""

    custom_id: str
    messages: list[dict[str, str]]
    sampling: SamplingConfig


class BatchResult(BaseModel):
    """Result from a batch request."""

    custom_id: str
    response: str
    error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class BatchClient:
    """Client for OpenAI Batch API operations."""

    def __init__(self, config: ClientConfig | None = None):
        self.config = config or ClientConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    async def create_batch_file(
        self,
        requests: list[BatchRequest],
        output_path: str | Path | None = None,
    ) -> str:
        """Create a JSONL batch file for upload.

        Args:
            requests: List of batch requests.
            output_path: Optional path to save the file.

        Returns:
            JSONL content as string.
        """
        lines = []
        for req in requests:
            body: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": req.messages,
                **req.sampling.to_api_params(),
            }
            if self.config.enable_thinking:
                body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
            item = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(item, ensure_ascii=False))

        content = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(content, encoding="utf-8")
            logger.info(f"Batch file saved to {output_path}")

        return content

    async def submit_batch(
        self,
        requests: list[BatchRequest],
        description: str = "Benchmark batch",
    ) -> str:
        """Submit a batch for processing.

        Args:
            requests: List of batch requests.
            description: Description for the batch.

        Returns:
            Batch ID.
        """
        # Create batch file
        batch_content = await self.create_batch_file(requests)

        # Upload file
        file_obj = await self._client.files.create(
            file=batch_content.encode("utf-8"),
            purpose="batch",
        )
        logger.info(f"Uploaded batch file: {file_obj.id}")

        # Create batch
        batch = await self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        logger.info(f"Created batch: {batch.id}")

        return batch.id

    async def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
    ) -> str:
        """Wait for a batch to complete and return the output file ID.

        Args:
            batch_id: Batch ID to wait for.
            poll_interval: Seconds between status checks.
            timeout: Maximum time to wait in seconds.

        Returns:
            Output file ID.
        """
        start_time = time.time()

        while True:
            batch = await self._client.batches.retrieve(batch_id)
            status = batch.status
            logger.info(f"Batch {batch_id} status: {status}")

            if status == "completed":
                if batch.output_file_id:
                    return batch.output_file_id
                raise RuntimeError(f"Batch completed but no output file: {batch}")

            if status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} failed with status: {status}")

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {timeout}s"
                )

            await asyncio.sleep(poll_interval)

    async def get_batch_results(
        self,
        output_file_id: str,
    ) -> list[BatchResult]:
        """Download and parse batch results.

        Args:
            output_file_id: ID of the output file.

        Returns:
            List of batch results.
        """
        content = await self._client.files.content(output_file_id)
        text = content.content.decode("utf-8")

        results = []
        for line in text.strip().split("\n"):
            if not line:
                continue

            data = json.loads(line)
            custom_id = data["custom_id"]

            if data.get("error"):
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        response="",
                        error=data["error"].get("message", "Unknown error"),
                    )
                )
                continue

            response_body = data.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])

            if choices:
                message = choices[0].get("message", {})
                response_text = message.get("content", "")
                usage = response_body.get("usage", {})
            else:
                response_text = ""
                usage = {}

            results.append(
                BatchResult(
                    custom_id=custom_id,
                    response=response_text,
                    usage=usage,
                )
            )

        logger.info(f"Retrieved {len(results)} batch results")
        return results

    async def run_batch(
        self,
        requests: list[BatchRequest],
        description: str = "Benchmark batch",
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
    ) -> list[BatchResult]:
        """Submit a batch, wait for completion, and return results.

        This is a convenience method that combines submit_batch,
        wait_for_batch, and get_batch_results.

        Args:
            requests: List of batch requests.
            description: Description for the batch.
            poll_interval: Seconds between status checks.
            timeout: Maximum time to wait in seconds.

        Returns:
            List of batch results.
        """
        batch_id = await self.submit_batch(requests, description)
        output_file_id = await self.wait_for_batch(batch_id, poll_interval, timeout)
        return await self.get_batch_results(output_file_id)

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

        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": True}}
            if self.config.enable_thinking
            else None
        )

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

    async def test_connection(self) -> bool:
        """Test the connection to the API server.

        Returns:
            True if connection is successful.
        """
        try:
            response = await self.query_single("Hello", SamplingConfig(max_tokens=10))
            logger.info(f"Connection test successful: {response[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
