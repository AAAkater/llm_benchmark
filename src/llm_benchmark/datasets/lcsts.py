"""LCSTS dataset for Chinese short text summarization."""

import re
from typing import Literal

from datasets import load_dataset
from pydantic import validate_call

from llm_benchmark.datasets.base import BaseDataset, Sample
from llm_benchmark.utils import logger


class LCSTSDataset(BaseDataset):
    """LCSTS dataset for Chinese short text summarization."""

    name = "hugcyp/LCSTS"
    samples: list[Sample] = []

    @validate_call
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "test",
        data_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Load LCSTS dataset (internal method).

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

        # Load from Hugging Face
        dataset = load_dataset(
            data_dir or self.name,
            split=split,
            trust_remote_code=not data_dir,
        ).to_iterable_dataset()

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            # LCSTS has empty summaries in test set, use the text as reference
            summary: str = item["summary"] if item["summary"] else item["text"][:100]
            self.samples.append(
                Sample(
                    id=f"lcsts_{i}",
                    text=item["text"],
                    reference=summary,
                    metadata={"source": "lcsts"},
                )
            )

        logger.info(f"Loaded {len(self.samples)} LCSTS samples from {split} split")
        return

    def create_prompt(self, sample: Sample) -> str:
        """Create Chinese summarization prompt."""
        return f"""
        {sample.text}

        请为以上文本生成一个简短的摘要

        这些为参考示例
        文本: 12月12日,多家被立案稽查的沪市公司集体对外发布退市风险提示公告,*ST国创位列""黑名单""。目前证监会调查仍在进行,*ST国创尚未收到此次立案调查书面结论意见。一旦立案调查事项触及相关规定,公司股票将被实施退市风险警示。"
        摘要: 信披违规外加业绩亏损*ST国创退市风险概率大增

        文本: 据微信公众号“界面”报道,4日上午10点左右,中国发改委反垄断调查小组突击查访奔驰上海办事处,调取数据材料,并对多名奔驰高管进行了约谈。截止昨日晚9点,包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内
        摘要: 发改委反垄断调查小组突击调查奔驰上海办事处
        
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
        text = text.strip('",。！')
        return text

    def get_evaluator(self):
        """Get the ROUGE evaluator for Chinese text."""
        from llm_benchmark.evaluators.rouge import RougeChineseEvaluator

        return RougeChineseEvaluator()
