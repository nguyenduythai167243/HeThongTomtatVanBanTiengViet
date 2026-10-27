"""Mô hình đối chuẩn Baseline 1: Lead-k Extractive Summarizer.
Trong báo chí tiếng Việt (cấu trúc hình tháp ngược), các câu đầu tiên chứa đựng hầu hết thông tin cốt lõi (Who, What, Where, When, Why).
Lead-k chọn k câu đầu tiên làm bản tóm tắt đại diện.
"""

from typing import List, Dict, Any
from src.preprocessor import VietnamesePreprocessor

class LeadKSummarizer:
    """Bộ tóm tắt Lead-k: Trích xuất k câu đầu tiên của văn bản."""

    def __init__(self, k: int = 3):
        self.k = max(1, k)
        self.model_name = f"Lead-{self.k}"

    def summarize(self, text: str) -> str:
        """Tóm tắt văn bản bằng cách lấy k câu đầu tiên."""
        sentences = VietnamesePreprocessor.split_sentences(text)
        if not sentences:
            return ""
        selected = sentences[:self.k]
        return " ".join(selected).strip()

    def batch_summarize(self, texts: List[str]) -> List[str]:
        """Tóm tắt hàng loạt danh sách văn bản."""
        return [self.summarize(t) for t in texts]

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_type": "extractive_baseline",
            "model_name": self.model_name,
            "k": self.k,
            "description": f"Trích xuất {self.k} câu đầu tiên theo quy tắc cấu trúc hình tháp ngược."
        }
