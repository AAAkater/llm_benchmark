"""Base classes for dataset loaders."""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel


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
        split: str = "test",
        data_dir: str | Path | None = None,
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
