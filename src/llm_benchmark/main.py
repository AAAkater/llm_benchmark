"""Main benchmark runner for LLM evaluation."""

import json
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
from llm_benchmark.inference.base import TpsStats
from llm_benchmark.utils import logger


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    # Batch settings
    max_concurrent: int = 10  # 降低并发数，避免服务器过载

    # Output settings
    output_dir: str = "results"

    # Description for the benchmark run
    description: str = ""  # 描述信息，用于标识本次基准测试


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
                    "description": self.config.description,
                    "n_samples": len(self.samples),
                    **avg_scores_dict,
                    # TPS stats (shortened field names)
                    "in_tok": self.tps_stats.total_input_tokens,
                    "out_tok": self.tps_stats.total_output_tokens,
                    "min_in": self.tps_stats.min_input_tokens,
                    "max_in": self.tps_stats.max_input_tokens,
                    "min_out": self.tps_stats.min_output_tokens,
                    "max_out": self.tps_stats.max_output_tokens,
                    "lat_ms": self.tps_stats.avg_latency_ms,
                    "min_lat": self.tps_stats.min_latency_ms,
                    "max_lat": self.tps_stats.max_latency_ms,
                    "ttft_ms": self.tps_stats.avg_ttft_ms,
                    "min_ttft": self.tps_stats.min_ttft_ms,
                    "max_ttft": self.tps_stats.max_ttft_ms,
                    "avg_tps": self.tps_stats.avg_output_tps,
                    "min_tps": self.tps_stats.min_output_tps,
                    "max_tps": self.tps_stats.max_output_tps,
                    "ts": self.timestamp,
                }
            ]
        )


def save_benchmark_results(result: BenchmarkResult, output_dir: str | Path = "results") -> None:
    """Save benchmark results to files.

    Detailed results are saved as JSONL, summary is saved as CSV.

    Args:
        result: BenchmarkResult to save.
        output_dir: Directory to save results to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSONL
    # Replace / with _ to avoid path issues (e.g., "hugcyp/LCSTS" -> "hugcyp_LCSTS")
    safe_dataset_name = result.dataset_name.replace("/", "_")
    detail_path = output_dir / f"{safe_dataset_name}_{result.timestamp}.jsonl"

    rows: list[dict[str, Any]] = []
    for sample, pred, eval_result in zip(result.samples, result.predictions, result.evaluation_results):
        scores_dict = eval_result.scores.model_dump()
        row = {
            "sample_id": sample.id,
            "text": sample.text,
            "reference": sample.reference,
            "prediction": pred,
            **scores_dict,
            "dataset": result.dataset_name,
            "description": result.config.description,
            "timestamp": result.timestamp,
        }
        rows.append(row)

    with open(detail_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Saved detailed results to {detail_path}")

    # Append to summary file (CSV)
    summary_path = output_dir / "benchmark_summary.csv"
    summary_df = result.summary_dataframe()

    # Check if file exists to determine if we need header
    header = not summary_path.exists()
    summary_df.to_csv(summary_path, mode="a", header=header, index=False, encoding="utf-8")
    logger.info(f"Updated summary at {summary_path}")


class BenchmarkRunner:
    """Runner for LLM sampling parameter benchmarks."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        dataset: BaseDataset,
        enable_thinking: bool = False,
    ):
        self.client = OAIBatchClient(client, model_name, enable_thinking)
        self.dataset = dataset

    async def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a benchmark with the given configuration.

        Args:
            config: Benchmark configuration (sampling parameters).
                   If None, uses default BenchmarkConfig().

        Returns:
            BenchmarkResult with all evaluation data.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting benchmark: {self.dataset.name}")

        samples = self.dataset.samples

        # Create prompts
        prompts = [self.dataset.create_prompt(s) for s in samples]

        # Get predictions
        logger.info(f"Starting inference for {len(prompts)} prompts...")
        inference_results = await self.client.query_batch_concurrent(
            prompts=prompts,
            max_concurrent=config.max_concurrent,
        )
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
        return result
