"""Main benchmark runner for LLM sampling parameter evaluation."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from llm_benchmark.client import (
    BatchClient,
    BatchRequest,
    ClientConfig,
    SamplingConfig,
)
from llm_benchmark.datasets import (
    Sample,
    create_prompt_for_dataset,
    get_dataset,
    load_dataset_by_name,
)
from llm_benchmark.evaluation import (
    EvaluationResult,
    RougeEvaluator,
    create_evaluator_for_dataset,
)
from llm_benchmark.utils.logger import logger


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    # Dataset settings
    dataset_name: str
    split: str = "test"
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
    use_batch_api: bool = False  # 本地服务器通常不支持 Batch API
    batch_size: int = 100
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
            "use_batch_api": self.use_batch_api,
            "batch_size": self.batch_size,
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
        for i, (sample, pred, eval_result) in enumerate(
            zip(self.samples, self.predictions, self.evaluation_results)
        ):
            rows.append(
                {
                    "sample_id": sample.id,
                    "text": sample.text,
                    "reference": sample.reference,
                    "prediction": pred,
                    "rouge1": eval_result.rouge_scores.rouge1,
                    "rouge2": eval_result.rouge_scores.rouge2,
                    "rougeL": eval_result.rouge_scores.rougeL,
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


def create_prompt(sample: Sample, dataset_name: str) -> str:
    """Create a prompt for the model based on the dataset type.

    Args:
        sample: The sample to create a prompt for.
        dataset_name: Name of the dataset.

    Returns:
        Formatted prompt string.
    """
    return create_prompt_for_dataset(dataset_name, sample)


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

        if not await self.client.test_connection():
            logger.error("Failed to connect to API server")
            raise Exception("Failed to connect to API server")
        # Load dataset
        samples = load_dataset_by_name(
            name=config.dataset_name,
            split=config.split,
            data_dir=config.data_dir,
            max_samples=config.max_samples,
        )

        # Create prompts
        prompts = [create_prompt(s, config.dataset_name) for s in samples]

        # Get predictions
        logger.info(f"Starting inference for {len(prompts)} prompts...")
        if config.use_batch_api:
            predictions = await self._run_batch_inference(prompts, config)
        else:
            predictions = await self._run_concurrent_inference(prompts, config)
        logger.info(f"Inference completed, got {len(predictions)} predictions")

        # Postprocess predictions
        dataset = get_dataset(config.dataset_name)
        predictions = [dataset.postprocess(p) for p in predictions]
        logger.info(f"Applied postprocessing for {config.dataset_name}")

        # Evaluate
        evaluator = create_evaluator_for_dataset(config.dataset_name)
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

    async def _run_batch_inference(
        self,
        prompts: list[str],
        config: BenchmarkConfig,
    ) -> list[str]:
        """Run inference using Batch API."""
        sampling = config.to_sampling_config()
        all_predictions = [None] * len(prompts)

        # Process in batches
        for batch_start in tqdm(
            range(0, len(prompts), config.batch_size),
            desc="Processing batches",
        ):
            batch_end = min(batch_start + config.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Create batch requests
            requests = [
                BatchRequest(
                    custom_id=f"req_{batch_start + i}",
                    messages=[{"role": "user", "content": p}],
                    sampling=sampling,
                )
                for i, p in enumerate(batch_prompts)
            ]

            try:
                # Try Batch API
                results = await self.client.run_batch(
                    requests=requests,
                    description=f"Benchmark batch {batch_start}-{batch_end}",
                )

                for result in results:
                    idx = int(result.custom_id.split("_")[1])
                    all_predictions[idx] = result.response

            except Exception as e:
                logger.warning(f"Batch API failed, falling back to concurrent: {e}")
                # Fallback to concurrent requests
                batch_preds = await self.client.query_batch_concurrent(
                    prompts=batch_prompts,
                    sampling=sampling,
                    max_concurrent=config.max_concurrent,
                )
                for i, pred in enumerate(batch_preds):
                    all_predictions[batch_start + i] = pred

        return all_predictions

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
        detail_path = (
            self.output_dir / f"{result.config.dataset_name}_{result.timestamp}.jsonl"
        )
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


async def run_sampling_sweep(
    dataset_name: str,
    temperatures: list[float] | None = None,
    top_ps: list[float] | None = None,
    max_samples: int = 100,
    output_dir: str = "results",
    client_config: ClientConfig | None = None,
) -> list[BenchmarkResult]:
    """Run a sweep over sampling parameters.

    Args:
        dataset_name: Dataset to evaluate on.
        temperatures: List of temperatures to test.
        top_ps: List of top_p values to test.
        max_samples: Maximum samples per run.
        output_dir: Output directory for results.
        client_config: Optional client configuration.

    Returns:
        List of BenchmarkResult objects.
    """
    temperatures = temperatures or [0.0, 0.3, 0.7, 1.0]
    top_ps = top_ps or [0.8, 0.9, 0.95, 1.0]

    runner = BenchmarkRunner(client_config=client_config, output_dir=output_dir)
    results = []

    # Temperature sweep
    logger.info("Running temperature sweep...")
    for temp in temperatures:
        config = BenchmarkConfig(
            dataset_name=dataset_name,
            temperature=temp,
            max_samples=max_samples,
        )
        result = await runner.run(config)
        results.append(result)

    # Top-p sweep
    logger.info("Running top-p sweep...")
    for top_p in top_ps:
        config = BenchmarkConfig(
            dataset_name=dataset_name,
            top_p=top_p,
            max_samples=max_samples,
        )
        result = await runner.run(config)
        results.append(result)

    return results


async def main() -> None:
    """Main entry point for the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Sampling Parameter Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["lcsts", "xsum", "truthfulqa"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a parameter sweep instead of single config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:1234/v1",
        help="Base URL for the API server",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="local-model",
        help="Model name to use",
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(
        f"{args.output_dir}/benchmark.log",
        rotation="10 MB",
        level="DEBUG",
    )

    # Create client config
    client_config = ClientConfig(
        base_url=args.base_url,
        model_name=args.model_name,
    )

    # Test connection
    client = BatchClient(client_config)
    if not await client.test_connection():
        logger.error("Failed to connect to API server")
        return

    if args.sweep:
        await run_sampling_sweep(
            dataset_name=args.dataset,
            max_samples=args.max_samples or 100,
            output_dir=args.output_dir,
            client_config=client_config,
        )
    else:
        config = BenchmarkConfig(
            dataset_name=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )

        runner = BenchmarkRunner(
            client_config=client_config,
            output_dir=args.output_dir,
        )
        await runner.run(config)


if __name__ == "__main__":
    asyncio.run(main())
