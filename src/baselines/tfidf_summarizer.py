"""Mô hình đối chuẩn Baseline 2: TF-IDF Extractive Summarizer.
Phương pháp trích xuất câu dựa trên tần số xuất hiện của từ (Term Frequency) và nghịch đảo tần số văn bản (Inverse Document Frequency),
kết hợp hệ số vị trí câu (Position Prior) để phản ánh tầm quan trọng của các câu đầu bài viết.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessor import VietnamesePreprocessor

class TFIDFSummarizer:
    """Bộ tóm tắt trích xuất dựa trên TF-IDF và hệ số vị trí."""

    def __init__(
        self,
        num_sentences: int = 3,
        min_df: int = 1,
        max_df: float = 0.95,
        ngram_range: tuple = (1, 2),
        position_weight: float = 0.25
    ):
        self.num_sentences = max(1, num_sentences)
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.position_weight = position_weight
        self.model_name = f"TF-IDF-Extractive(top_{self.num_sentences})"

    def summarize(self, text: str) -> str:
        """Tóm tắt một văn bản bằng phương pháp xếp hạng câu TF-IDF."""
        sentences = VietnamesePreprocessor.split_sentences(text)
        n_sent = len(sentences)
        if n_sent == 0:
            return ""
        if n_sent <= self.num_sentences:
            return " ".join(sentences)

        # Xây dựng vectorizer TF-IDF trên các câu của chính văn bản
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=VietnamesePreprocessor.tokenize_syllables,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                token_pattern=None
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # Điểm TF-IDF trung bình của các từ trong câu
            raw_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            # Chuẩn hóa về [0, 1]
            max_score = np.max(raw_scores) if np.max(raw_scores) > 0 else 1.0
            norm_scores = raw_scores / max_score
        except Exception:
            # Fallback nếu số lượng từ vựng quá ít
            norm_scores = np.ones(n_sent)

        # Kết hợp hệ số vị trí (Position prior: câu đầu tiên có điểm vị trí cao nhất)
        position_factors = np.array([1.0 / (idx + 1) for idx in range(n_sent)])
        final_scores = (1 - self.position_weight) * norm_scores + self.position_weight * position_factors

        # Chọn top_k câu có điểm cao nhất
        top_indices = sorted(np.argsort(final_scores)[-self.num_sentences:])

        # Sắp xếp các câu được chọn theo thứ tự xuất hiện ban đầu để văn bản mạch lạc
        selected_sentences = [sentences[i] for i in sorted(top_indices)]
        return " ".join(selected_sentences).strip()

    def batch_summarize(self, texts: List[str]) -> List[str]:
        """Tóm tắt hàng loạt danh sách văn bản."""
        return [self.summarize(t) for t in texts]

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_type": "extractive_tfidf",
            "model_name": self.model_name,
            "num_sentences": self.num_sentences,
            "position_weight": self.position_weight,
            "description": "Trích xuất câu theo tổng trọng số TF-IDF kết hợp hệ số suy giảm vị trí (position prior)."
        }
