"""BLEU evaluation metrics for translation and code generation tasks.

Supports:
- BLEU score calculation
- Used for Flores, Summscreen, Govrepcrs, Iwdlt2017, etc.
"""

from collections import Counter
from math import exp, log
from typing import override

from llm_benchmark.evaluators.base import BaseEvaluator, BaseScores, EvaluationResult
from llm_benchmark.utils import logger


class BleuScores(BaseScores):
    """BLEU scores for a single sample."""

    bleu: float
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    brevity_penalty: float


class BleuEvaluator(BaseEvaluator[BleuScores]):
    """Evaluator for BLEU metrics.

    Used for machine translation and code generation tasks.
    """

    def __init__(
        self,
        max_n: int = 4,
        language: str = "en",
    ):
        """Initialize the evaluator.

        Args:
            max_n: Maximum n-gram order (default 4 for BLEU-4).
            language: Language for tokenization ("en" or "zh").
        """
        self.max_n = max_n
        self.language = language
        logger.info(f"Using BLEU evaluator (max_n={max_n}) for {language}")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        if not text:
            return []

        if self.language == "zh":
            # For Chinese, use jieba
            import jieba

            return list(jieba.cut(text))
        else:
            # For English, simple whitespace tokenization
            return text.split()

    def _get_ngrams(
        self,
        tokens: list[str],
        n: int,
    ) -> Counter:
        """Get n-grams from tokens.

        Args:
            tokens: List of tokens.
            n: N-gram order.

        Returns:
            Counter of n-grams.
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] += 1
        return ngrams

    def _compute_bleu(
        self,
        prediction_tokens: list[str],
        reference_tokens: list[str],
    ) -> tuple[float, list[float], float]:
        """Compute BLEU score.

        Args:
            prediction_tokens: Tokenized prediction.
            reference_tokens: Tokenized reference.

        Returns:
            Tuple of (bleu, [bleu1, bleu2, bleu3, bleu4], brevity_penalty).
        """
        if not prediction_tokens or not reference_tokens:
            return 0.0, [0.0, 0.0, 0.0, 0.0], 0.0

        # Compute brevity penalty
        pred_len = len(prediction_tokens)
        ref_len = len(reference_tokens)

        if pred_len >= ref_len:
            brevity_penalty = 1.0
        elif pred_len == 0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = exp(1 - ref_len / pred_len)

        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._get_ngrams(prediction_tokens, n)
            ref_ngrams = self._get_ngrams(reference_tokens, n)

            # Count clipped matches
            matches = 0
            total_pred = 0

            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
                total_pred += count

            if total_pred == 0:
                precision = 0.0
            else:
                precision = matches / total_pred

            precisions.append(precision)

        # Compute BLEU score
        if all(p > 0 for p in precisions):
            log_precision = sum(log(p) for p in precisions) / len(precisions)
            bleu = brevity_penalty * exp(log_precision)
        else:
            bleu = 0.0

        # Pad precisions to 4 values
        while len(precisions) < 4:
            precisions.append(0.0)

        return bleu, precisions[:4], brevity_penalty

    @override
    def compute_scores(
        self,
        prediction: str,
        reference: str,
    ) -> BleuScores:
        """Compute BLEU for a single prediction.

        Args:
            prediction: Model prediction.
            reference: Reference text.

        Returns:
            BleuScores object with BLEU scores.
        """
        if not prediction or not reference:
            return BleuScores(
                bleu=0.0,
                bleu1=0.0,
                bleu2=0.0,
                bleu3=0.0,
                bleu4=0.0,
                brevity_penalty=0.0,
            )

        # Tokenize
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        # Compute BLEU
        bleu, precisions, bp = self._compute_bleu(pred_tokens, ref_tokens)

        return BleuScores(
            bleu=bleu,
            bleu1=precisions[0],
            bleu2=precisions[1],
            bleu3=precisions[2],
            bleu4=precisions[3],
            brevity_penalty=bp,
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
            references: List of reference texts.
            sample_ids: Optional list of sample IDs.

        Returns:
            Tuple of (list of EvaluationResult, average scores dict).
        """
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch: {len(predictions)} predictions, {len(references)} references")

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(predictions))]

        results = []
        total_bleu = 0.0
        total_bleu1 = 0.0
        total_bleu2 = 0.0
        total_bleu3 = 0.0
        total_bleu4 = 0.0

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            scores = self.compute_scores(pred, ref)
            total_bleu += scores.bleu
            total_bleu1 += scores.bleu1
            total_bleu2 += scores.bleu2
            total_bleu3 += scores.bleu3
            total_bleu4 += scores.bleu4

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
            "bleu": total_bleu / n if n > 0 else 0.0,
            "bleu1": total_bleu1 / n if n > 0 else 0.0,
            "bleu2": total_bleu2 / n if n > 0 else 0.0,
            "bleu3": total_bleu3 / n if n > 0 else 0.0,
            "bleu4": total_bleu4 / n if n > 0 else 0.0,
        }

        logger.info(
            f"BLEU: {avg_scores['bleu']:.4f}, BLEU-1: {avg_scores['bleu1']:.4f}, BLEU-4: {avg_scores['bleu4']:.4f}"
        )

        return results, avg_scores
