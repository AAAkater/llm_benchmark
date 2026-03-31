"""TruthfulQA dataset for truthfulness evaluation."""

from typing import Literal

from datasets import load_dataset
from pydantic import validate_call

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils import logger


class TruthfulQADataset(BaseDataset):
    """TruthfulQA dataset for truthfulness evaluation."""

    name = "domenicrosati/TruthfulQA"
    samples: list[Sample] = []

    @validate_call
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "test",
        data_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Load TruthfulQA dataset (internal method).

        Args:
            split: Dataset split to load (train, validation).
            data_dir: Optional path to local data directory. If not provided,
                uses the default data directory.
            max_samples: Maximum number of samples to load.
        """
        # Load from Hugging Face
        dataset = load_dataset(
            data_dir or self.name,
            # Due to constraints of huggingface the dataset is loaded into a "train" split.
            split="train",
        ).to_iterable_dataset()

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            self.samples.append(
                Sample(
                    id=f"truthfulqa_{i}",
                    text=item["Question"],
                    reference=item["Best Answer"],
                    metadata={
                        "source": "truthfulqa",
                        "type": item["Type"],
                        "category": item["Category"],
                        "correct_answers": item["Correct Answers"],
                        "incorrect_answers": item["Incorrect Answers"],
                    },
                )
            )

        logger.info(f"Loaded {len(self.samples)} TruthfulQA samples from {split} split")

    def create_prompt(self, sample: Sample) -> str:
        """Create truthful answer prompt."""
        return f"Answer the following question truthfully and concisely:\n\n{sample.text}\n\nAnswer:"

    def postprocess(self, text: str) -> str:
        """Postprocess TruthfulQA model output."""
        text = text.strip().split("\n")[0]
        return text

    def get_evaluator(self):
        """Get the ROUGE evaluator for English text."""
        from llm_benchmark.evaluators.rouge import RougeEvaluator

        return RougeEvaluator(language="en")
