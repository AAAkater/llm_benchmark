"""Inference clients for LLM benchmarking."""

from llm_benchmark.inference.oai_compatible import (
    InferenceResult,
    OAIBatchClient,
)

__all__ = [
    "OAIBatchClient",
    "InferenceResult",
]
