"""Base classes for evaluators."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel


class BaseScores(BaseModel):
    """Base class for evaluation scores."""

    pass


ScoresT = TypeVar("ScoresT", bound=BaseScores)


class EvaluationResult(BaseModel):
    """Complete evaluation result for a sample."""

    sample_id: str
    prediction: str
    reference: str
    scores: BaseScores
    metadata: dict[str, str] | None = None


class BaseEvaluator(ABC, Generic[ScoresT]):
    """Abstract base class for evaluators.

    Each evaluator implementation should inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def compute_scores(
        self,
        prediction: str,
        reference: str,
    ) -> ScoresT:
        """Compute scores for a single prediction.

        Args:
            prediction: Generated text.
            reference: Reference text.

        Returns:
            Scores object.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sample_ids: list[str] | None = None,
    ) -> tuple[list[EvaluationResult], dict[str, float]]:
        """Evaluate a batch of predictions.

        Args:
            predictions: List of generated texts.
            references: List of reference texts.
            sample_ids: Optional list of sample IDs.

        Returns:
            Tuple of (list of EvaluationResult, average scores dict).
        """
        ...
