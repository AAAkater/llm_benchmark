from pydantic import BaseModel

from llm_benchmark.inference import InferenceResult


class TpsStats(BaseModel):
    """TPS statistics for a benchmark run."""

    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    min_input_tokens: int = 0
    max_input_tokens: int = 0
    min_output_tokens: int = 0
    max_output_tokens: int = 0

    # Latency
    total_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # TPS
    avg_output_tps: float = 0.0
    avg_total_tps: float = 0.0
    min_output_tps: float = 0.0
    max_output_tps: float = 0.0

    @classmethod
    def from_results(cls, results: list[InferenceResult]) -> "TpsStats":
        """Calculate TPS statistics from inference results."""
        if not results:
            return cls()

        input_tokens = [r.input_tokens for r in results]
        output_tokens = [r.output_tokens for r in results]
        latencies = [r.latency_ms for r in results]
        output_tps = [r.output_tps for r in results if r.output_tps > 0]
        total_tps = [r.total_tps for r in results if r.total_tps > 0]

        total_input = sum(input_tokens)
        total_output = sum(output_tokens)
        total_latency = sum(latencies)
        n = len(results)

        return cls(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            min_input_tokens=min(input_tokens),
            max_input_tokens=max(input_tokens),
            min_output_tokens=min(output_tokens),
            max_output_tokens=max(output_tokens),
            total_latency_ms=total_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_latency_ms=total_latency / n,
            avg_output_tps=sum(output_tps) / len(output_tps) if output_tps else 0.0,
            avg_total_tps=sum(total_tps) / len(total_tps) if total_tps else 0.0,
            min_output_tps=min(output_tps) if output_tps else 0.0,
            max_output_tps=max(output_tps) if output_tps else 0.0,
        )
