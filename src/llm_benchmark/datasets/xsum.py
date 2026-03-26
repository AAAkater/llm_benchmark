"""XSum dataset for English news summarization."""

from pathlib import Path

from datasets import load_dataset

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils.logger import logger


class XSumDataset(BaseDataset):
    """XSum dataset for English news summarization."""

    name = "xsum"
    default_data_dir = ""

    def load(
        self,
        split: str = "test",
        data_dir: str | Path | None = None,
        max_samples: int | None = None,
    ) -> list[Sample]:
        """Load XSum dataset."""
        data_dir = data_dir or self.default_data_dir
        dataset = load_dataset(
            "xsum",
            split=split,
            data_dir=str(data_dir),
            trust_remote_code=True,
        )

        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            samples.append(
                Sample(
                    id=f"xsum_{item.get('id', i)}",
                    text=item["document"],
                    reference=item["summary"],
                    metadata={"source": "xsum"},
                )
            )

        logger.info(f"Loaded {len(samples)} XSum samples from {split} split")
        return samples

    def create_prompt(self, sample: Sample) -> str:
        """Create English summarization prompt."""
        return f"Summarize the following article in one sentence:\n\n{sample.text}\n\nSummary:"

    def postprocess(self, text: str) -> str:
        """Postprocess XSum model output."""
        text = text.strip().split("\n")[0]
        text = text.replace("1. ", "") if text.startswith("1. ") else text
        text = text.replace("- ", "") if text.startswith("- ") else text
        return text
