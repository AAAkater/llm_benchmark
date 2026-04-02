"""CLI interface for LLM benchmark runner."""

import argparse
import asyncio

from openai import AsyncOpenAI

from llm_benchmark.datasets import get_dataset
from llm_benchmark.main import BenchmarkConfig, BenchmarkRunner
from llm_benchmark.utils import logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    return parser.parse_args()


async def run_benchmark() -> None:
    """Main entry point for the benchmark CLI."""
    args = parse_args()

    # Configure logging
    logger.add(
        f"{args.output_dir}/benchmark.log",
        rotation="10 MB",
        level="DEBUG",
    )

    # Create OpenAI client
    client = AsyncOpenAI(base_url=args.base_url, api_key="sk-dummy")

    # Load dataset
    dataset = get_dataset(
        name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )

    # Create benchmark config
    config = BenchmarkConfig()

    runner = BenchmarkRunner(
        client=client,
        model_name=args.model_name,
        dataset=dataset,
        output_dir=args.output_dir,
    )
    await runner.run(config)


def main() -> None:
    """Synchronous entry point for CLI."""
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
