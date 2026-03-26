"""LCSTS dataset for Chinese short text summarization."""

import re
from typing import Literal

from datasets import load_dataset

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils.logger import logger


class LCSTSDataset(BaseDataset):
    """LCSTS dataset for Chinese short text summarization."""

    name = "hugcyp/LCSTS"

    def load(
        self,
        split: Literal["train", "validation", "test"] = "test",
        data_dir: str | None = None,
        max_samples: int | None = None,
    ) -> list[Sample]:
        """Load LCSTS dataset.

        Args:
            split: Dataset split to load (train, validation, test).
            data_dir: Optional path to local data directory. If not provided,
                dataset will be downloaded from Hugging Face.
            max_samples: Maximum number of samples to load.

        Returns:
            List of Sample objects.
        """

        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Unknown split: {split}. Available: {['train', 'validation', 'test']}")

        samples: list[Sample] = []

        # Load from Hugging Face
        dataset = load_dataset(
            data_dir or self.name,
            split=split,
            trust_remote_code=not data_dir,
        )

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            # LCSTS has empty summaries in test set, use the text as reference
            summary = item.get("summary", "") or item.get("text", "")[:100]
            samples.append(
                Sample(
                    id=f"lcsts_{i}",
                    text=item["text"],
                    reference=summary,
                    metadata={"source": "lcsts"},
                )
            )

        logger.info(f"Loaded {len(samples)} LCSTS samples from {split} split")
        return samples

    def create_prompt(self, sample: Sample) -> str:
        """Create Chinese summarization prompt."""
        return f"""
        {sample.text}

        请为以上文本生成一个简短的摘要
        
        必须将你最后的答案放在<answer></answer>标签中,我们会提取最后一组answer标签作为你的答案
        """

    def postprocess(self, text: str) -> str:
        """Postprocess LCSTS model output.

        Extracts content from <answer></answer> tags and cleans up artifacts.
        """

        # Extract content from <answer></answer> tags
        # Find all matches and take the last non-empty one
        matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not matches:
            logger.debug(f"Output (last 100 chars): ...{text[-200:]}")
            raise ValueError("No <answer></answer> tags found in output")
        # Take the last non-empty match
        for match in reversed(matches):
            if match.strip():
                text = match
                break
        else:
            logger.debug(f"Output (last 100 chars): ...{text[-200:]}")
            raise ValueError("All <answer></answer> tags are empty in output")
        # Clean up common artifacts
        text = text.strip()
        text = text.replace("1. ", "") if text.startswith("1. ") else text
        text = text.replace("- ", "") if text.startswith("- ") else text
        text = text.strip('"，。！')
        return text
