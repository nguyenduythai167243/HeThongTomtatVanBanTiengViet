"""Module đánh giá chỉ số ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) với tokenization tiếng Việt cố định.
Tuân thủ nghiêm ngặt quy định mục 13:
- Tokenization cố định chuẩn âm tiết tiếng Việt.
- Xử lý các trường hợp rỗng / lỗi: gán điểm 0.0 thay vì loại bỏ mẫu để thổi phồng điểm số.
- Báo cáo đầy đủ Precision, Recall và F1-Score.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from src.preprocessor import VietnamesePreprocessor

class ROUGEEvaluator:
    """Bộ tính điểm ROUGE chuẩn mực cho bài toán tóm tắt tiếng Việt."""

    def __init__(self):
        # Kiểm tra xem rouge_score có sẵn không
        try:
            from rouge_score import rouge_scorer
            # Sử dụng custom tokenizer cố định tiếng Việt cho rouge_scorer
            self.has_rouge_pkg = True
            self.scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=False
            )
        except Exception:
            self.has_rouge_pkg = False

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Trích xuất từ điển n-gram và số lần xuất hiện."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    @classmethod
    def compute_rouge_n(
        cls, hyp_tokens: List[str], ref_tokens: List[str], n: int
    ) -> Tuple[float, float, float]:
        """Tính ROUGE-N (Precision, Recall, F1)."""
        if not hyp_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0

        hyp_ngrams = cls._get_ngrams(hyp_tokens, n)
        ref_ngrams = cls._get_ngrams(ref_tokens, n)

        total_ref_ngrams = max(1, len(ref_tokens) - n + 1)
        total_hyp_ngrams = max(1, len(hyp_tokens) - n + 1)

        overlapping_count = 0
        for ngram, count in hyp_ngrams.items():
            if ngram in ref_ngrams:
                overlapping_count += min(count, ref_ngrams[ngram])

        recall = overlapping_count / total_ref_ngrams
        precision = overlapping_count / total_hyp_ngrams
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @classmethod
    def compute_rouge_l(
        cls, hyp_tokens: List[str], ref_tokens: List[str]
    ) -> Tuple[float, float, float]:
        """Tính ROUGE-L dựa trên Dãy con chung dài nhất (Longest Common Subsequence - LCS)."""
        m, n = len(hyp_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return 0.0, 0.0, 0.0

        # Ma trận quy hoạch động LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if hyp_tokens[i] == ref_tokens[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

        lcs_len = dp[m][n]
        recall = lcs_len / n
        precision = lcs_len / m
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def evaluate_single(self, hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
        """Đánh giá 1 cặp (tóm tắt sinh ra vs tóm tắt vàng).
        Nếu bản tóm tắt rỗng, trả về 0.0 tuyệt đối (không drop).
        """
        hyp_tokens = VietnamesePreprocessor.tokenize_syllables(hypothesis.lower())
        ref_tokens = VietnamesePreprocessor.tokenize_syllables(reference.lower())

        p1, r1, f1_1 = self.compute_rouge_n(hyp_tokens, ref_tokens, 1)
        p2, r2, f1_2 = self.compute_rouge_n(hyp_tokens, ref_tokens, 2)
        pl, rl, f1_l = self.compute_rouge_l(hyp_tokens, ref_tokens)

        return {
            "rouge-1": {"precision": round(p1, 4), "recall": round(r1, 4), "f1": round(f1_1, 4)},
            "rouge-2": {"precision": round(p2, 4), "recall": round(r2, 4), "f1": round(f1_2, 4)},
            "rouge-l": {"precision": round(pl, 4), "recall": round(rl, 4), "f1": round(f1_l, 4)}
        }

    def evaluate_corpus(
        self, hypotheses: List[str], references: List[str]
    ) -> Dict[str, Any]:
        """Đánh giá toàn bộ tập dữ liệu test và báo cáo điểm trung bình."""
        assert len(hypotheses) == len(references), "Số lượng giả thuyết và nhãn tham chiếu phải bằng nhau!"
        n_samples = len(hypotheses)
        if n_samples == 0:
            return {"num_samples": 0}

        r1_p, r1_r, r1_f = [], [], []
        r2_p, r2_r, r2_f = [], [], []
        rl_p, rl_r, rl_f = [], [], []

        for hyp, ref in zip(hypotheses, references):
            scores = self.evaluate_single(hyp, ref)
            r1_p.append(scores["rouge-1"]["precision"])
            r1_r.append(scores["rouge-1"]["recall"])
            r1_f.append(scores["rouge-1"]["f1"])

            r2_p.append(scores["rouge-2"]["precision"])
            r2_r.append(scores["rouge-2"]["recall"])
            r2_f.append(scores["rouge-2"]["f1"])

            rl_p.append(scores["rouge-l"]["precision"])
            rl_r.append(scores["rouge-l"]["recall"])
            rl_f.append(scores["rouge-l"]["f1"])

        return {
            "num_samples": n_samples,
            "rouge-1": {
                "precision": round(float(np.mean(r1_p)), 4),
                "recall": round(float(np.mean(r1_r)), 4),
                "f1": round(float(np.mean(r1_f)), 4),
                "f1_std": round(float(np.std(r1_f)), 4)
            },
            "rouge-2": {
                "precision": round(float(np.mean(r2_p)), 4),
                "recall": round(float(np.mean(r2_r)), 4),
                "f1": round(float(np.mean(r2_f)), 4),
                "f1_std": round(float(np.std(r2_f)), 4)
            },
            "rouge-l": {
                "precision": round(float(np.mean(rl_p)), 4),
                "recall": round(float(np.mean(rl_r)), 4),
                "f1": round(float(np.mean(rl_f)), 4),
                "f1_std": round(float(np.std(rl_f)), 4)
            }
        }
