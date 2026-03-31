"""XSum dataset for English news summarization."""

from typing import Literal

from datasets import load_dataset
from pydantic import validate_call

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils import logger


class XSumDataset(BaseDataset):
    """XSum dataset for English news summarization."""

    name = "EdinburghNLP/xsum"
    samples: list[Sample] = []

    @validate_call
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "test",
        data_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Load XSum dataset (internal method).

        Args:
            split: Dataset split to load (train, validation, test).
            data_dir: Optional path to local data directory. If not provided,
                dataset will be downloaded from Hugging Face.
            max_samples: Maximum number of samples to load.
        """
        # Load from Hugging Face
        dataset = load_dataset(
            data_dir or self.name,
            split=split,
        ).to_iterable_dataset()

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            self.samples.append(
                Sample(
                    id=f"xsum_{item.get('id', i)}",
                    text=item["document"],
                    reference=item["summary"],
                    metadata={"source": "xsum"},
                )
            )

        logger.info(f"Loaded {len(self.samples)} XSum samples from {split} split")

    def create_prompt(self, sample: Sample) -> str:
        """Create English summarization prompt."""
        return f"Summarize the following article in one sentence:\n\n{sample.text}\n\nSummary:"

    def postprocess(self, text: str) -> str:
        """Postprocess XSum model output."""
        text = text.strip().split("\n")[0]
        text = text.replace("1. ", "") if text.startswith("1. ") else text
        text = text.replace("- ", "") if text.startswith("- ") else text
        return text

    def get_evaluator(self):
        """Get the ROUGE evaluator for English text."""
        from llm_benchmark.evaluators.rouge import RougeEvaluator

        return RougeEvaluator(language="en")
