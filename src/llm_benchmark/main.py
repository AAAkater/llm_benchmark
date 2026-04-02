"""Main benchmark runner for LLM evaluation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel

from llm_benchmark.datasets import (
    BaseDataset,
    Sample,
)
from llm_benchmark.evaluators.rouge import EvaluationResult
from llm_benchmark.inference import (
    InferenceResult,
    OAIBatchClient,
)
from llm_benchmark.utils import logger


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    # Batch settings
    max_concurrent: int = 10  # 降低并发数，避免服务器过载

    # Output settings
    output_dir: str = "results"


class TpsStats(BaseModel):
    """TPS statistics for a benchmark run."""

    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    min_input_tokens: int = 0
    max_input_tokens: int = 0
    min_output_tokens: int = 0
    max_output_tokens: int = 0

    # Latency
    total_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # TPS
    avg_output_tps: float = 0.0
    avg_total_tps: float = 0.0
    min_output_tps: float = 0.0
    max_output_tps: float = 0.0

    @classmethod
    def from_results(cls, results: list[InferenceResult]) -> "TpsStats":
        """Calculate TPS statistics from inference results."""
        if not results:
            return cls()

        input_tokens = [r.input_tokens for r in results]
        output_tokens = [r.output_tokens for r in results]
        latencies = [r.latency_ms for r in results]
        output_tps = [r.output_tps for r in results if r.output_tps > 0]
        total_tps = [r.total_tps for r in results if r.total_tps > 0]

        total_input = sum(input_tokens)
        total_output = sum(output_tokens)
        total_latency = sum(latencies)
        n = len(results)

        return cls(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            min_input_tokens=min(input_tokens),
            max_input_tokens=max(input_tokens),
            min_output_tokens=min(output_tokens),
            max_output_tokens=max(output_tokens),
            total_latency_ms=total_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_latency_ms=total_latency / n,
            avg_output_tps=sum(output_tps) / len(output_tps) if output_tps else 0.0,
            avg_total_tps=sum(total_tps) / len(total_tps) if total_tps else 0.0,
            min_output_tps=min(output_tps) if output_tps else 0.0,
            max_output_tps=max(output_tps) if output_tps else 0.0,
        )


class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""

    dataset_name: str
    config: BenchmarkConfig
    samples: list[Sample]
    predictions: list[str]
    evaluation_results: list[EvaluationResult]
    avg_scores: dict[str, float]
    tps_stats: TpsStats
    timestamp: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        rows: list[dict[str, Any]] = []
        for sample, pred, eval_result in zip(self.samples, self.predictions, self.evaluation_results):
            scores_dict = eval_result.scores.model_dump()
            rows.append(
                {
                    "sample_id": sample.id,
                    "text": sample.text,
                    "reference": sample.reference,
                    "prediction": pred,
                    **scores_dict,
                    "dataset": self.dataset_name,
                    "timestamp": self.timestamp,
                }
            )
        return pd.DataFrame(rows)

    def summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame with average scores."""
        # Dynamically include all average scores with "avg_" prefix
        avg_scores_dict = {f"avg_{k}": v for k, v in self.avg_scores.items()}
        return pd.DataFrame(
            [
                {
                    "dataset": self.dataset_name,
                    "n_samples": len(self.samples),
                    **avg_scores_dict,
                    # TPS stats
                    "total_input_tokens": self.tps_stats.total_input_tokens,
                    "total_output_tokens": self.tps_stats.total_output_tokens,
                    "min_input_tokens": self.tps_stats.min_input_tokens,
                    "max_input_tokens": self.tps_stats.max_input_tokens,
                    "min_output_tokens": self.tps_stats.min_output_tokens,
                    "max_output_tokens": self.tps_stats.max_output_tokens,
                    "avg_latency_ms": self.tps_stats.avg_latency_ms,
                    "avg_output_tps": self.tps_stats.avg_output_tps,
                    "min_output_tps": self.tps_stats.min_output_tps,
                    "max_output_tps": self.tps_stats.max_output_tps,
                    "timestamp": self.timestamp,
                }
            ]
        )


class BenchmarkRunner:
    """Runner for LLM sampling parameter benchmarks."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        dataset: BaseDataset,
        enable_thinking: bool = False,
        output_dir: str = "results",
    ):
        self.client = OAIBatchClient(client, model_name, enable_thinking)
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, config: BenchmarkConfig | None = None) -> BenchmarkResult:
        """Run a benchmark with the given configuration.

        Args:
            config: Benchmark configuration (sampling parameters).
                   If None, uses default BenchmarkConfig().

        Returns:
            BenchmarkResult with all evaluation data.
        """
        if config is None:
            config = BenchmarkConfig()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting benchmark: {self.dataset.name}")

        samples = self.dataset.samples

        # Create prompts
        prompts = [self.dataset.create_prompt(s) for s in samples]

        # Get predictions
        logger.info(f"Starting inference for {len(prompts)} prompts...")
        inference_results = await self._run_concurrent_inference(prompts, config)
        logger.info(f"Inference completed, got {len(inference_results)} predictions")

        # Calculate TPS stats
        tps_stats = TpsStats.from_results(inference_results)
        logger.info(
            f"TPS Stats: avg_output_tps={tps_stats.avg_output_tps:.2f}, "
            f"total_input={tps_stats.total_input_tokens}, total_output={tps_stats.total_output_tokens}"
        )

        # Postprocess predictions
        processed_predictions: list[str] = []
        processed_samples: list[Sample] = []
        processed_results: list[InferenceResult] = []
        for sample, inf_result in zip(samples, inference_results):
            try:
                processed_pred = self.dataset.postprocess(inf_result.response)
                processed_predictions.append(processed_pred)
                processed_samples.append(sample)
                processed_results.append(inf_result)
            except Exception as e:
                logger.warning(f"Failed to postprocess sample {sample.id}: {e}")
                continue
        predictions = processed_predictions
        samples = processed_samples
        # Recalculate TPS stats for processed samples only
        tps_stats = TpsStats.from_results(processed_results)
        logger.info(f"Applied postprocessing for {self.dataset.name}, {len(predictions)} samples retained")

        # Evaluate
        evaluator = self.dataset.get_evaluator()
        eval_results, avg_scores = evaluator.evaluate(
            predictions=predictions,
            references=[s.reference for s in samples],
            sample_ids=[s.id for s in samples],
        )

        result = BenchmarkResult(
            dataset_name=self.dataset.name,
            config=config,
            samples=samples,
            predictions=predictions,
            evaluation_results=eval_results,
            avg_scores=avg_scores,
            tps_stats=tps_stats,
            timestamp=timestamp,
        )

        # Save results
        self._save_results(result)

        return result

    async def _run_concurrent_inference(
        self,
        prompts: list[str],
        config: BenchmarkConfig,
    ) -> list[InferenceResult]:
        """Run inference using concurrent async requests."""

        predictions = await self.client.query_batch_concurrent(
            prompts=prompts,
            max_concurrent=config.max_concurrent,
        )

        return predictions

    def _save_results(self, result: BenchmarkResult) -> None:
        """Save benchmark results to files in JSONL format."""
        import json

        # Save detailed results
        df = result.to_dataframe()
        # Replace / with _ to avoid path issues (e.g., "hugcyp/LCSTS" -> "hugcyp_LCSTS")
        safe_dataset_name = result.dataset_name.replace("/", "_")
        detail_path = self.output_dir / f"{safe_dataset_name}_{result.timestamp}.jsonl"
        with open(detail_path, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved detailed results to {detail_path}")

        # Append to summary file
        summary_path = self.output_dir / "benchmark_summary.jsonl"
        summary_df = result.summary_dataframe()

        with open(summary_path, "a", encoding="utf-8") as f:
            for record in summary_df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Updated summary at {summary_path}")
