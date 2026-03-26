"""Dataset loaders for summarization benchmarks.

Supports:
- LCSTS: Chinese short text summarization
- Xsum: English news summarization
- TruthfulQA: Truthfulness evaluation (generation mode)
"""

from .base import BaseDataset, Sample
from .lcsts import LCSTSDataset
from .truthfulqa import TruthfulQADataset
from .xsum import XSumDataset

# Dataset registry
DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    "lcsts": LCSTSDataset,
    "xsum": XSumDataset,
    "truthfulqa": TruthfulQADataset,
}


def get_dataset(name: str) -> BaseDataset:
    """Get a dataset instance by name.

    Args:
        name: Dataset name (lcsts, xsum, truthfulqa).

    Returns:
        Dataset instance.

    Raises:
        ValueError: If dataset name is unknown.
    """
    name_lower = name.lower()
    if name_lower not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name_lower]()


def load_dataset_by_name(
    name: str,
    split: str = "test",
    data_dir: str | None = None,
    max_samples: int | None = None,
) -> list[Sample]:
    """Load a dataset by name.

    Args:
        name: Dataset name (lcsts, xsum, truthfulqa).
        split: Dataset split to load.
        data_dir: Optional custom data directory.
        max_samples: Maximum number of samples to load.

    Returns:
        List of Sample objects.
    """
    dataset = get_dataset(name)
    return dataset.load(split=split, data_dir=data_dir, max_samples=max_samples)


def create_prompt_for_dataset(name: str, sample: Sample) -> str:
    """Create a prompt for a dataset sample.

    Args:
        name: Dataset name.
        sample: The sample to create a prompt for.

    Returns:
        Formatted prompt string.
    """
    dataset = get_dataset(name)
    return dataset.create_prompt(sample)


def postprocess_output(name: str, text: str) -> str:
    """Postprocess model output for a dataset.

    Args:
        name: Dataset name.
        text: Raw model output.

    Returns:
        Cleaned output text.
    """
    dataset = get_dataset(name)
    return dataset.postprocess(text)


# Backward compatibility - keep old function names
DATASET_LOADERS = {name: ds().load for name, ds in DATASET_REGISTRY.items()}
DATASET_POSTPROCESSORS = {
    name: ds().postprocess for name, ds in DATASET_REGISTRY.items()
}

__all__ = [
    "Sample",
    "BaseDataset",
    "LCSTSDataset",
    "XSumDataset",
    "TruthfulQADataset",
    "DATASET_REGISTRY",
    "get_dataset",
    "load_dataset_by_name",
    "create_prompt_for_dataset",
    "postprocess_output",
    "DATASET_LOADERS",
    "DATASET_POSTPROCESSORS",
]
