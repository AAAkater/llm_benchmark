"""Dataset loaders for summarization benchmarks.

Supports:
- LCSTS: Chinese short text summarization
- Xsum: English news summarization
- TruthfulQA: Truthfulness evaluation (generation mode)
"""

from typing import Literal

from .base import BaseDataset, Sample
from .lcsts import LCSTSDataset
from .truthfulqa import TruthfulQADataset
from .xsum import XSumDataset

# Dataset registry
DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    LCSTSDataset.name: LCSTSDataset,
    XSumDataset.name: XSumDataset,
    TruthfulQADataset.name: TruthfulQADataset,
}


def get_dataset(
    name: str,
    split: Literal["train", "validation", "test"] = "test",
    data_dir: str | None = None,
    max_samples: int | None = None,
) -> BaseDataset:
    """Get a dataset instance by name.

    Args:
        name: Dataset name (lcsts, xsum, truthfulqa).
        split: Dataset split to load.
        data_dir: Optional custom data directory.
        max_samples: Maximum number of samples to load.

    Returns:
        Dataset instance with loaded samples.

    Raises:
        ValueError: If dataset name is unknown.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](split=split, data_dir=data_dir, max_samples=max_samples)


__all__ = [
    "Sample",
    "BaseDataset",
    "LCSTSDataset",
    "XSumDataset",
    "TruthfulQADataset",
    "DATASET_REGISTRY",
    "get_dataset",
]
