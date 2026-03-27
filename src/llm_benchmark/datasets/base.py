"""Base classes for dataset loaders."""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from llm_benchmark.evaluators.base import BaseEvaluator


class Sample(BaseModel):
    """A single sample for evaluation."""

    id: str
    text: str
    reference: str
    metadata: dict[str, str] | None = None


class BaseDataset(ABC):
    """Abstract base class for dataset loaders.

    Each dataset implementation should inherit from this class and implement
    the required abstract methods.
    """

    name: str = "base"

    @abstractmethod
    def load(
        self,
        split: Literal["train", "validation", "test"] = "test",
        data_dir: str | None = None,
        max_samples: int | None = None,
    ) -> list[Sample]:
        """Load dataset samples.

        Args:
            split: Dataset split to load.
            data_dir: Path to the data directory.
            max_samples: Maximum number of samples to load.

        Returns:
            List of Sample objects.
        """
        ...

    @abstractmethod
    def create_prompt(self, sample: Sample) -> str:
        """Create a prompt for the model.

        Args:
            sample: The sample to create a prompt for.

        Returns:
            Formatted prompt string.
        """
        ...

    @abstractmethod
    def postprocess(self, text: str) -> str:
        """Postprocess model output.

        Args:
            text: Raw model output.

        Returns:
            Cleaned output text.
        """
        ...

    @abstractmethod
    def get_evaluator(self) -> BaseEvaluator:
        """Get the appropriate evaluator for this dataset.

        Returns:
            BaseEvaluator configured for the dataset's language.
        """
        ...
