"""ROUGE evaluation metrics for summarization tasks.

Supports:
- ROUGE-1, ROUGE-2, ROUGE-L (via rouge-score)
- Chinese ROUGE (via rouge-chinese)
"""

from typing import Literal, override

from rouge_chinese import Rouge as ChineseRouge
from rouge_score import rouge_scorer

from llm_benchmark.evaluators.base import BaseEvaluator, BaseScores, EvaluationResult
from llm_benchmark.utils.logger import logger


class RougeScores(BaseScores):
    """ROUGE scores for a single sample."""

    rouge1: float
    rouge2: float
    rougeL: float


class RougeEvaluator(BaseEvaluator[RougeScores]):
    """Evaluator for ROUGE metrics."""

    def __init__(
        self,
        language: Literal["en", "zh"] = "en",
        use_stemmer: bool = True,
    ):
        """Initialize the evaluator.

        Args:
            language: Language for tokenization ("en" or "zh").
            use_stemmer: Whether to use stemming for English.
        """
        self.language = language
        self.use_stemmer = use_stemmer

        if language == "zh":
            self._chinese_rouge = ChineseRouge()
            self._use_chinese = True
            logger.info("Using rouge-chinese for Chinese text")
        else:
            self._scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=use_stemmer,
            )
            self._use_chinese = False
            logger.info("Using rouge-score for evaluation")

    def _tokenize_chinese(self, text: str) -> str:
        """Tokenize Chinese text using jieba."""
        import jieba

        return " ".join(jieba.cut(text))

    @override
    def compute_scores(
        self,
        prediction: str,
        reference: str,
    ) -> RougeScores:
        """Compute ROUGE scores for a single prediction.

        Args:
            prediction: Generated summary.
            reference: Reference summary.

        Returns:
            RougeScores object with rouge1, rouge2, rougeL.
        """
        if not prediction or not reference:
            return RougeScores(rouge1=0.0, rouge2=0.0, rougeL=0.0)

        if self._use_chinese:
            # Tokenize for Chinese
            pred_tokens = self._tokenize_chinese(prediction)
            ref_tokens = self._tokenize_chinese(reference)

            scores = self._chinese_rouge.get_scores(pred_tokens, ref_tokens)[0]

            return RougeScores(
                rouge1=scores["rouge-1"]["f"],
                rouge2=scores["rouge-2"]["f"],
                rougeL=scores["rouge-l"]["f"],
            )
        else:
            scores = self._scorer.score(reference, prediction)

            return RougeScores(
                rouge1=scores["rouge1"].fmeasure,
                rouge2=scores["rouge2"].fmeasure,
                rougeL=scores["rougeL"].fmeasure,
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
            predictions: List of generated summaries.
            references: List of reference summaries.
            sample_ids: Optional list of sample IDs.

        Returns:
            Tuple of (list of EvaluationResult, average scores dict).
        """
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch: {len(predictions)} predictions, {len(references)} references")

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(predictions))]

        results = []
        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            scores = self.compute_scores(pred, ref)
            total_rouge1 += scores.rouge1
            total_rouge2 += scores.rouge2
            total_rougeL += scores.rougeL

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
            "rouge1": total_rouge1 / n if n > 0 else 0.0,
            "rouge2": total_rouge2 / n if n > 0 else 0.0,
            "rougeL": total_rougeL / n if n > 0 else 0.0,
        }

        logger.info(
            f"Average ROUGE scores: R1={avg_scores['rouge1']:.4f}, "
            f"R2={avg_scores['rouge2']:.4f}, RL={avg_scores['rougeL']:.4f}"
        )

        return results, avg_scores
