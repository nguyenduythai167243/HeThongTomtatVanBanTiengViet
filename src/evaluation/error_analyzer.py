"""Module phân tích lỗi chuyên sâu (Error Analysis):
Tập trung vào 3 nhóm lỗi trọng tâm của Đề tài NLP-20:
1. Thiếu ý (Omission / Incompleteness): Tóm tắt bỏ sót thông điệp quan trọng của bài gốc (ROUGE Recall thấp).
2. Bịa thông tin (Factuality / Hallucination): Tự suy diễn hoặc sinh ra số liệu, tên riêng sai lệch không có trong bài gốc.
3. Thất bại trên văn bản dài (Long-Document Failure / Truncation Bias): Suy giảm chất lượng khi văn bản vượt ngưỡng ngữ cảnh.
"""

from typing import List, Dict, Any
from src.evaluation.rouge_evaluator import ROUGEEvaluator
from src.evaluation.fact_checker import FactualityChecker
from src.preprocessor import VietnamesePreprocessor

class SummarizationErrorAnalyzer:
    """Bộ phân tích hồ sơ lỗi và thống kê các trường hợp thất bại của mô hình."""

    def __init__(self):
        self.rouge_evaluator = ROUGEEvaluator()

    def analyze_sample(
        self,
        sample_id: str,
        document: str,
        reference: str,
        hypothesis: str,
        domain: str = "Chung"
    ) -> Dict[str, Any]:
        """Phân tích chi tiết lỗi của một mẫu dự đoán cụ thể."""
        doc_words = len(VietnamesePreprocessor.tokenize_syllables(document))
        hyp_words = len(VietnamesePreprocessor.tokenize_syllables(hypothesis))
        ref_words = len(VietnamesePreprocessor.tokenize_syllables(reference))

        # 1. Đo lường chỉ số ROUGE
        rouge_scores = self.rouge_evaluator.evaluate_single(hypothesis, reference)
        r1_recall = rouge_scores["rouge-1"]["recall"]
        r1_f1 = rouge_scores["rouge-1"]["f1"]

        # 2. Kiểm tra tính chân thực (bịa thông tin)
        fact_res = FactualityChecker.check_factuality(hypothesis, document)

        # 3. Phân loại 3 nhóm lỗi chính
        errors_detected = []

        # Nhóm 1: Thiếu ý (Omission)
        # Tiêu chí: ROUGE-1 Recall thấp (< 0.35) hoặc độ dài tóm tắt quá ngắn so với tham chiếu (< 40%)
        is_omission = (r1_recall < 0.35) or (hyp_words < 0.4 * ref_words)
        if is_omission:
            errors_detected.append({
                "category": "THIEU_Y",
                "description": f"Thiếu ý cốt lõi (ROUGE-1 Recall={r1_recall:.2f}, tỷ lệ độ dài={hyp_words}/{ref_words} từ).",
                "severity": "HIGH" if r1_recall < 0.20 else "MEDIUM"
            })

        # Nhóm 2: Bịa thông tin (Hallucination)
        # Tiêu chí: Có số liệu hoặc tên riêng không tồn tại trong bài gốc
        if fact_res["has_hallucination"]:
            errors_detected.append({
                "category": "BIA_THONG_TIN",
                "description": (
                    f"Ảo giác dữ kiện: Sai lệch số liệu ({fact_res['numeric_consistency']['hallucinated']}) "
                    f"hoặc thực thể tên riêng ({fact_res['entity_consistency']['hallucinated']})."
                ),
                "severity": "CRITICAL"
            })

        # Nhóm 3: Lỗi văn bản dài (Long text failure)
        # Tiêu chí: Văn bản gốc dài (> 800 từ) nhưng tóm tắt chỉ phản ánh câu đầu tiên, bỏ quên toàn bộ kết quả phía sau
        is_long_doc_issue = (doc_words >= 800) and (r1_f1 < 0.30)
        if is_long_doc_issue:
            errors_detected.append({
                "category": "VAN_BAN_DAI",
                "description": f"Thất bại trên văn bản dài ({doc_words} từ): Bị cắt cụt ngữ cảnh (Truncation bias) dẫn đến mất kết luận cuối bài.",
                "severity": "HIGH"
            })

        return {
            "id": sample_id,
            "domain": domain,
            "doc_length": doc_words,
            "hyp_length": hyp_words,
            "ref_length": ref_words,
            "rouge_scores": rouge_scores,
            "factuality": fact_res,
            "errors": errors_detected,
            "has_error": len(errors_detected) > 0
        }

    def analyze_corpus_errors(
        self, samples: List[Dict[str, Any]], predictions: List[str]
    ) -> Dict[str, Any]:
        """Thống kê hồ sơ lỗi trên toàn bộ tập dữ liệu test."""
        total_samples = len(samples)
        error_counts = {
            "THIEU_Y": 0,
            "BIA_THONG_TIN": 0,
            "VAN_BAN_DAI": 0,
            "NO_SIGNIFICANT_ERROR": 0
        }
        detailed_error_profiles = []

        for sample, pred in zip(samples, predictions):
            analysis = self.analyze_sample(
                sample_id=sample.get("id", "unknown"),
                document=sample["document"],
                reference=sample["summary"],
                hypothesis=pred,
                domain=sample.get("domain", "Chung")
            )
            detailed_error_profiles.append(analysis)

            if not analysis["has_error"]:
                error_counts["NO_SIGNIFICANT_ERROR"] += 1
            else:
                for err in analysis["errors"]:
                    cat = err["category"]
                    if cat in error_counts:
                        error_counts[cat] += 1

        return {
            "total_evaluated_samples": total_samples,
            "error_distribution": error_counts,
            "error_percentages": {
                k: round((v / total_samples) * 100, 2) if total_samples > 0 else 0.0
                for k, v in error_counts.items()
            },
            "sample_profiles": detailed_error_profiles
        }
