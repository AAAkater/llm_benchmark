"""Main benchmark runner for LLM sampling parameter evaluation."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from llm_benchmark.client import (
    BatchClient,
    ClientConfig,
    SamplingConfig,
)
from llm_benchmark.datasets import (
    Sample,
    create_prompt_for_dataset,
    get_dataset,
    load_dataset_by_name,
)
from llm_benchmark.evaluators.rouge import EvaluationResult, RougeScores
from llm_benchmark.utils.logger import logger


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    # Dataset settings
    dataset_name: str
    split: Literal["train", "validation", "test"] = "test"
    data_dir: str | None = None
    max_samples: int | None = None

    # Sampling parameters to test
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 256
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Batch settings
    max_concurrent: int = 10  # 降低并发数，避免服务器过载

    # Output settings
    output_dir: str = "results"

    def to_sampling_config(self) -> SamplingConfig:
        """Convert to SamplingConfig."""
        return SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/saving."""
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "max_samples": self.max_samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""

    config: BenchmarkConfig
    samples: list[Sample]
    predictions: list[str]
    evaluation_results: list[EvaluationResult]
    avg_scores: dict[str, float]
    timestamp: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        rows = []
        for i, (sample, pred, eval_result) in enumerate(zip(self.samples, self.predictions, self.evaluation_results)):
            scores = RougeScores.model_validate(eval_result.scores)
            rows.append(
                {
                    "sample_id": sample.id,
                    "text": sample.text,
                    "reference": sample.reference,
                    "prediction": pred,
                    "rouge1": scores.rouge1,
                    "rouge2": scores.rouge2,
                    "rougeL": scores.rougeL,
                    **self.config.to_dict(),
                    "timestamp": self.timestamp,
                }
            )
        return pd.DataFrame(rows)

    def summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame with average scores."""
        return pd.DataFrame(
            [
                {
                    "dataset": self.config.dataset_name,
                    "split": self.config.split,
                    "n_samples": len(self.samples),
                    "avg_rouge1": self.avg_scores["rouge1"],
                    "avg_rouge2": self.avg_scores["rouge2"],
                    "avg_rougeL": self.avg_scores["rougeL"],
                    **self.config.to_dict(),
                    "timestamp": self.timestamp,
                }
            ]
        )


class BenchmarkRunner:
    """Runner for LLM sampling parameter benchmarks."""

    def __init__(
        self,
        client_config: ClientConfig | None = None,
        output_dir: str = "results",
    ):
        self.client = BatchClient(client_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a benchmark with the given configuration.

        Args:
            config: Benchmark configuration.

        Returns:
            BenchmarkResult with all evaluation data.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting benchmark: {config.dataset_name}")
        logger.info(f"Config: {config.to_dict()}")

        # Load dataset
        samples = load_dataset_by_name(
            name=config.dataset_name,
            split=config.split,
            data_dir=config.data_dir,
            max_samples=config.max_samples,
        )

        # Create prompts
        prompts = [create_prompt_for_dataset(config.dataset_name, s) for s in samples]

        # Get predictions
        logger.info(f"Starting inference for {len(prompts)} prompts...")
        predictions = await self._run_concurrent_inference(prompts, config)
        logger.info(f"Inference completed, got {len(predictions)} predictions")

        # Postprocess predictions
        dataset = get_dataset(config.dataset_name)
        processed_predictions = []
        processed_samples = []
        for sample, pred in zip(samples, predictions):
            try:
                processed_pred = dataset.postprocess(pred)
                processed_predictions.append(processed_pred)
                processed_samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to postprocess sample {sample.id}: {e}")
                continue
        predictions = processed_predictions
        samples = processed_samples
        logger.info(f"Applied postprocessing for {config.dataset_name}, {len(predictions)} samples retained")

        # Evaluate
        evaluator = dataset.get_evaluator()
        eval_results, avg_scores = evaluator.evaluate(
            predictions=predictions,
            references=[s.reference for s in samples],
            sample_ids=[s.id for s in samples],
        )

        result = BenchmarkResult(
            config=config,
            samples=samples,
            predictions=predictions,
            evaluation_results=eval_results,
            avg_scores=avg_scores,
            timestamp=timestamp,
        )

        # Save results
        self._save_results(result)

        return result

    async def _run_concurrent_inference(
        self,
        prompts: list[str],
        config: BenchmarkConfig,
    ) -> list[str]:
        """Run inference using concurrent async requests."""
        sampling = config.to_sampling_config()

        predictions = await self.client.query_batch_concurrent(
            prompts=prompts,
            sampling=sampling,
            max_concurrent=config.max_concurrent,
        )

        return predictions

    def _save_results(self, result: BenchmarkResult) -> None:
        """Save benchmark results to files in JSONL format."""
        import json

        # Save detailed results
        df = result.to_dataframe()
        # Replace / with _ to avoid path issues (e.g., "hugcyp/LCSTS" -> "hugcyp_LCSTS")
        safe_dataset_name = result.config.dataset_name.replace("/", "_")
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
