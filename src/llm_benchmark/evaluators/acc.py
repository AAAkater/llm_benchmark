"""Accuracy evaluation metrics for classification and multiple-choice tasks.

Supports:
- Accuracy calculation for classification tasks
- Multiple-choice evaluation (MMLU, CEval, etc.)
"""

from typing import override

from llm_benchmark.evaluators.base import BaseEvaluator, BaseScores, EvaluationResult
from llm_benchmark.utils.logger import logger


class AccScores(BaseScores):
    """Accuracy scores for a single sample."""

    correct: bool
    prediction: str
    answer: str


class AccEvaluator(BaseEvaluator[AccScores]):
    """Evaluator for accuracy metrics.

    Used for classification tasks, multiple-choice questions, and true/false tasks.
    """

    def __init__(
        self,
        normalize: bool = True,
        strip_whitespace: bool = True,
    ):
        """Initialize the evaluator.

        Args:
            normalize: Whether to normalize text (lowercase, strip punctuation).
            strip_whitespace: Whether to strip whitespace from predictions.
        """
        self.normalize = normalize
        self.strip_whitespace = strip_whitespace
        logger.info("Using accuracy evaluator")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        if self.strip_whitespace:
            text = text.strip()

        if self.normalize:
            text = text.lower()
            # Remove common punctuation
            for char in ".,!?;:'\"()[]{}":
                text = text.replace(char, "")

        return text

    @override
    def compute_scores(
        self,
        prediction: str,
        reference: str,
    ) -> AccScores:
        """Compute accuracy for a single prediction.

        Args:
            prediction: Model prediction (e.g., "A", "B", "C", "D" or "True", "False").
            reference: Reference answer.

        Returns:
            AccScores object with correct status.
        """
        if not prediction or not reference:
            return AccScores(correct=False, prediction=prediction, answer=reference)

        pred_normalized = self._normalize_text(prediction)
        ref_normalized = self._normalize_text(reference)

        correct = pred_normalized == ref_normalized

        return AccScores(
            correct=correct,
            prediction=prediction,
            answer=reference,
        )

    @override
    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sample_ids: list[str] | None = None,
    ) -> tuple[list[EvaluationResult], dict[str, float]]:
        """Evaluate a batch of predictions.

        Args:
            predictions: List of model predictions.
            references: List of reference answers.
            sample_ids: Optional list of sample IDs.

        Returns:
            Tuple of (list of EvaluationResult, average scores dict).
        """
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch: {len(predictions)} predictions, {len(references)} references")

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(predictions))]

        results = []
        correct_count = 0

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            scores = self.compute_scores(pred, ref)
            if scores.correct:
                correct_count += 1

            results.append(
                EvaluationResult(
                    sample_id=sample_ids[i],
                    prediction=pred,
                    reference=ref,
                    scores=scores,
                )
            )

        n = len(predictions)
        accuracy = correct_count / n if n > 0 else 0.0

        avg_scores = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": n,
        }

        logger.info(f"Accuracy: {accuracy:.4f} ({correct_count}/{n})")

        return results, avg_scores
