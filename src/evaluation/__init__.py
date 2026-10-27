"""Module đánh giá chỉ số: ROUGE, Fact-checking (Hallucination) và Error Analysis."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.rouge_evaluator import ROUGEEvaluator
    from src.evaluation.fact_checker import FactualityChecker
    from src.evaluation.error_analyzer import SummarizationErrorAnalyzer

__all__ = ["ROUGEEvaluator", "FactualityChecker", "SummarizationErrorAnalyzer"]


def __getattr__(name: str):
    if name == "ROUGEEvaluator":
        from src.evaluation.rouge_evaluator import ROUGEEvaluator

        return ROUGEEvaluator
    if name == "FactualityChecker":
        from src.evaluation.fact_checker import FactualityChecker

        return FactualityChecker
    if name == "SummarizationErrorAnalyzer":
        from src.evaluation.error_analyzer import SummarizationErrorAnalyzer

        return SummarizationErrorAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
