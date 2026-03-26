"""LCSTS dataset for Chinese short text summarization."""

import json
import re
from pathlib import Path

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils.logger import logger


class LCSTSDataset(BaseDataset):
    """LCSTS dataset for Chinese short text summarization."""

    name = "hugcyp/LCSTS"
    default_data_dir = ""

    def load(
        self,
        split: str = "test",
        data_dir: str | Path | None = None,
        max_samples: int | None = None,
    ) -> list[Sample]:
        """Load LCSTS dataset."""
        data_path = Path(data_dir or self.default_data_dir)
        file_map = {
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "test": "test_public.jsonl",
        }

        if split not in file_map:
            raise ValueError(f"Unknown split: {split}. Available: {list(file_map.keys())}")

        file_path = data_path / file_map[split]
        if not file_path.exists():
            raise FileNotFoundError(f"LCSTS file not found: {file_path}")

        samples = []
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                data = json.loads(line)
                # LCSTS has empty summaries in test set, use the text as reference
                summary = data.get("summary", "") or data.get("text", "")[:100]
                samples.append(
                    Sample(
                        id=f"lcsts_{i}",
                        text=data["text"],
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
