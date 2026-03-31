"""Main benchmark runner for LLM sampling parameter evaluation."""

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
from llm_benchmark.evaluators.rouge import EvaluationResult, RougeScores
from llm_benchmark.inference import (
    BatchClient,
    SamplingConfig,
)
from llm_benchmark.utils.logger import logger


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run - sampling parameters only."""

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
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""

    dataset_name: str
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
                    "dataset": self.dataset_name,
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
                    "dataset": self.dataset_name,
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
        client: AsyncOpenAI,
        model_name: str,
        dataset: BaseDataset,
        enable_thinking: bool = False,
        output_dir: str = "results",
    ):
        self.client = BatchClient(client, model_name, enable_thinking)
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
        logger.info(f"Config: {config.to_dict()}")

        samples = self.dataset.samples

        # Create prompts
        prompts = [self.dataset.create_prompt(s) for s in samples]

        # Get predictions
        logger.info(f"Starting inference for {len(prompts)} prompts...")
        predictions = await self._run_concurrent_inference(prompts, config)
        logger.info(f"Inference completed, got {len(predictions)} predictions")

        # Postprocess predictions
        processed_predictions = []
        processed_samples = []
        for sample, pred in zip(samples, predictions):
            try:
                processed_pred = self.dataset.postprocess(pred)
                processed_predictions.append(processed_pred)
                processed_samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to postprocess sample {sample.id}: {e}")
                continue
        predictions = processed_predictions
        samples = processed_samples
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
