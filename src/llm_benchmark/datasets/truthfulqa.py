"""TruthfulQA dataset for truthfulness evaluation."""

import csv
from pathlib import Path

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils.logger import logger


class TruthfulQADataset(BaseDataset):
    """TruthfulQA dataset for truthfulness evaluation."""

    name = "truthfulqa"
    default_data_dir = ""

    def load(
        self,
        split: str = "validation",
        data_dir: str | Path | None = None,
        max_samples: int | None = None,
    ) -> list[Sample]:
        """Load TruthfulQA dataset.

        Note: TruthfulQA is typically used for multiple-choice evaluation,
        but here we use it for generation mode where the model generates answers.
        """
        data_path = Path(data_dir or self.default_data_dir)
        file_map = {
            "validation": "TruthfulQA.csv",
            "train": "train.csv",
        }

        if split not in file_map:
            raise ValueError(
                f"Unknown split: {split}. Available: {list(file_map.keys())}"
            )

        file_path = data_path / file_map[split]
        if not file_path.exists():
            raise FileNotFoundError(f"TruthfulQA file not found: {file_path}")

        samples = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                samples.append(
                    Sample(
                        id=f"truthfulqa_{i}",
                        text=row["Question"],
                        reference=row["Best Answer"],
                        metadata={
                            "source": "truthfulqa",
                            "type": row.get("Type", ""),
                            "category": row.get("Category", ""),
                            "correct_answers": row.get("Correct Answers", ""),
                            "incorrect_answers": row.get("Incorrect Answers", ""),
                        },
                    )
                )

        logger.info(f"Loaded {len(samples)} TruthfulQA samples from {split} split")
        return samples

    def create_prompt(self, sample: Sample) -> str:
        """Create truthful answer prompt."""
        return f"Answer the following question truthfully and concisely:\n\n{sample.text}\n\nAnswer:"

    def postprocess(self, text: str) -> str:
        """Postprocess TruthfulQA model output."""
        text = text.strip().split("\n")[0]
        return text
