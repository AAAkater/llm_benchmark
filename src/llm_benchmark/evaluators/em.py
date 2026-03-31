"""Exact Match evaluation metrics for QA and reading comprehension tasks.

Supports:
- Exact Match (EM) calculation
- F1 score for token overlap
- Used for CLUE_CMRC, CLUE_DRCD, DROP, SQuAD, etc.
"""

import re
from collections import Counter
from typing import override

from llm_benchmark.evaluators.base import BaseEvaluator, BaseScores, EvaluationResult
from llm_benchmark.utils import logger


class EMScores(BaseScores):
    """Exact Match scores for a single sample."""

    exact_match: bool
    f1: float
    precision: float
    recall: float


class EMEvaluator(BaseEvaluator[EMScores]):
    """Evaluator for Exact Match metrics.

    Used for question answering and reading comprehension tasks.
    """

    def __init__(
        self,
        normalize: bool = True,
        language: str = "en",
    ):
        """Initialize the evaluator.

        Args:
            normalize: Whether to normalize text before comparison.
            language: Language for tokenization ("en" or "zh").
        """
        self.normalize = normalize
        self.language = language
        logger.info(f"Using Exact Match evaluator for {language}")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        if self.language == "zh":
            # For Chinese, use jieba
            import jieba

            return list(jieba.cut(text))
        else:
            # For English, simple whitespace tokenization
            return text.split()

    def _compute_f1(
        self,
        prediction_tokens: list[str],
        reference_tokens: list[str],
    ) -> tuple[float, float, float]:
        """Compute F1, precision, and recall.

        Args:
            prediction_tokens: Tokenized prediction.
            reference_tokens: Tokenized reference.

        Returns:
            Tuple of (f1, precision, recall).
        """
        if not prediction_tokens and not reference_tokens:
            return 1.0, 1.0, 1.0

        if not prediction_tokens or not reference_tokens:
            return 0.0, 0.0, 0.0

        pred_counter = Counter(prediction_tokens)
        ref_counter = Counter(reference_tokens)

        common = sum((pred_counter & ref_counter).values())

        precision = common / len(prediction_tokens) if prediction_tokens else 0.0
        recall = common / len(reference_tokens) if reference_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1, precision, recall

    @override
    def compute_scores(
        self,
        prediction: str,
        reference: str,
    ) -> EMScores:
        """Compute Exact Match and F1 for a single prediction.

        Args:
            prediction: Model prediction.
            reference: Reference answer.

        Returns:
            EMScores object with EM, F1, precision, and recall.
        """
        if not prediction or not reference:
            return EMScores(exact_match=False, f1=0.0, precision=0.0, recall=0.0)

        # Normalize texts
        pred_normalized = self._normalize_text(prediction)
        ref_normalized = self._normalize_text(reference)

        # Compute exact match
        exact_match = pred_normalized == ref_normalized

        # Tokenize for F1
        pred_tokens = self._tokenize(pred_normalized)
        ref_tokens = self._tokenize(ref_normalized)

        # Compute F1
        f1, precision, recall = self._compute_f1(pred_tokens, ref_tokens)

        return EMScores(
            exact_match=exact_match,
            f1=f1,
            precision=precision,
            recall=recall,
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
        total_em = 0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            scores = self.compute_scores(pred, ref)
            total_em += 1 if scores.exact_match else 0
            total_f1 += scores.f1
            total_precision += scores.precision
            total_recall += scores.recall

            results.append(
                EvaluationResult(
                    sample_id=sample_ids[i],
                    prediction=pred,
                    reference=ref,
                    scores=scores,
                )
            )

        n = len(predictions)
        avg_scores = {
            "exact_match": total_em / n if n > 0 else 0.0,
            "f1": total_f1 / n if n > 0 else 0.0,
            "precision": total_precision / n if n > 0 else 0.0,
            "recall": total_recall / n if n > 0 else 0.0,
        }

        logger.info(f"Exact Match: {avg_scores['exact_match']:.4f}, F1: {avg_scores['f1']:.4f}")

        return results, avg_scores
