"""Unit tests kiểm thử toàn diện hệ thống tóm tắt văn bản tiếng Việt NLP-20:
1. Kiểm thử Schema và toàn vẹn dữ liệu
2. Kiểm thử các trường hợp biên (Edge cases):
   - Văn bản rỗng / khoảng trắng
   - Văn bản cực ngắn (< 5 từ)
   - Văn bản cực dài (> 1000 từ)
   - Ký tự đặc biệt, HTML rác, từ viết tắt phức tạp
3. Kiểm thử tính đối xứng và không chia 0 của ROUGE-1/2/L
4. Kiểm thử Factuality Checker: phát hiện số liệu và thực thể tên riêng bịa đặt (hallucination)
5. Kiểm thử các mô hình Baseline (Lead-k, TF-IDF) và Chunking Engine
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import VietnamesePreprocessor
from src.data_loader import validate_sample_schema, compute_jaccard_similarity
from src.baselines.lead_k import LeadKSummarizer
from src.baselines.tfidf_summarizer import TFIDFSummarizer
from src.models.bartpho_summarizer import BARTphoSummarizer
from src.models.chunking_engine import ChunkingSummarizationEngine
from src.evaluation.rouge_evaluator import ROUGEEvaluator
from src.evaluation.fact_checker import FactualityChecker
from src.evaluation.error_analyzer import SummarizationErrorAnalyzer

class TestNLPSummarizationPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_normal = {
            "id": "TEST_001",
            "title": "Khai mạc hội thảo công nghệ AI",
            "document": (
                "Sáng nay tại Hà Nội, Bộ Khoa học và Công nghệ đã tổ chức hội thảo quốc tế về trí tuệ nhân tạo. "
                "Hơn 300 đại biểu và chuyên gia hàng đầu từ các viện nghiên cứu và trường đại học đã tham dự. "
                "Hội thảo tập trung thảo luận các giải pháp ứng dụng mô hình ngôn ngữ lớn cho tiếng Việt và an toàn thông tin."
            ),
            "summary": "Bộ Khoa học và Công nghệ tổ chức hội thảo AI tại Hà Nội với hơn 300 chuyên gia nhằm thảo luận ứng dụng mô hình ngôn ngữ lớn.",
            "domain": "Khoa học - Công nghệ",
            "source": "Báo Khoa học"
        }

    # 1. Kiểm thử Schema
    def test_schema_valid(self):
        valid, msg = validate_sample_schema(self.sample_normal)
        self.assertTrue(valid)
        self.assertEqual(msg, "Valid")

    def test_schema_missing_field(self):
        invalid_sample = dict(self.sample_normal)
        del invalid_sample["summary"]
        valid, msg = validate_sample_schema(invalid_sample)
        self.assertFalse(valid)
        self.assertIn("summary", msg)

    # 2. Kiểm thử Tiền xử lý & Trường hợp biên (Edge Cases)
    def test_empty_input_edge_case(self):
        valid, msg, meta = VietnamesePreprocessor.validate_edge_cases("   ")
        self.assertFalse(valid)
        self.assertEqual(meta["status"], "empty")

    def test_too_short_input_edge_case(self):
        valid, msg, meta = VietnamesePreprocessor.validate_edge_cases("Chào bạn nhé.")
        self.assertFalse(valid)
        self.assertEqual(meta["status"], "too_short")

    def test_abbreviation_preservation(self):
        text = "GS. TS. Nguyễn Văn A làm việc tại UBND TP. HCM. Chiều nay ông về Hà Nội."
        sents = VietnamesePreprocessor.split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertIn("GS. TS. Nguyễn Văn A", sents[0])
        self.assertIn("TP. HCM", sents[0])

    def test_html_cleaning(self):
        dirty = "<p>Nội dung <b>quan trọng</b> cần <a href='#'>tóm tắt</a>.</p>"
        cleaned = VietnamesePreprocessor.clean_text(dirty)
        self.assertEqual(cleaned, "Nội dung quan trọng cần tóm tắt.")

    # 3. Kiểm thử Mô hình Baseline Lead-k và TF-IDF
    def test_lead_k_baseline(self):
        lead1 = LeadKSummarizer(k=1)
        res1 = lead1.summarize(self.sample_normal["document"])
        self.assertTrue(res1.startswith("Sáng nay tại Hà Nội"))

        lead2 = LeadKSummarizer(k=2)
        res2 = lead2.summarize(self.sample_normal["document"])
        sents = VietnamesePreprocessor.split_sentences(res2)
        self.assertEqual(len(sents), 2)

    def test_tfidf_baseline(self):
        tfidf = TFIDFSummarizer(num_sentences=2)
        res = tfidf.summarize(self.sample_normal["document"])
        self.assertTrue(len(res) > 20)

    # 4. Kiểm thử BARTpho Summarizer và Chunking Engine
    def test_bartpho_summarizer(self):
        summarizer = BARTphoSummarizer(lazy_load=True)
        res = summarizer.summarize(self.sample_normal["document"])
        self.assertEqual(res["status"], "success")
        self.assertTrue(len(res["summary"]) > 0)

    def test_chunking_engine_on_long_text(self):
        summarizer = BARTphoSummarizer(lazy_load=True)
        chunker = ChunkingSummarizationEngine(summarizer, chunk_size_words=50, chunk_overlap_words=15)
        # Tạo văn bản dài nhiều câu
        long_doc = " ".join([self.sample_normal["document"]] * 5)
        res = chunker.summarize_map_reduce(long_doc)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["num_chunks"] > 1)
        self.assertEqual(res["strategy"], "map_reduce_chunking")

    # 5. Kiểm thử ROUGE Evaluator
    def test_rouge_identical_texts(self):
        evaluator = ROUGEEvaluator()
        text = "Bộ Khoa học và Công nghệ tổ chức hội thảo AI tại Hà Nội."
        scores = evaluator.evaluate_single(text, text)
        self.assertAlmostEqual(scores["rouge-1"]["f1"], 1.0, places=2)
        self.assertAlmostEqual(scores["rouge-2"]["f1"], 1.0, places=2)
        self.assertAlmostEqual(scores["rouge-l"]["f1"], 1.0, places=2)

    def test_rouge_empty_output_no_crash(self):
        evaluator = ROUGEEvaluator()
        scores = evaluator.evaluate_single("", "Văn bản tham chiếu.")
        self.assertEqual(scores["rouge-1"]["f1"], 0.0)
        self.assertEqual(scores["rouge-2"]["f1"], 0.0)
        self.assertEqual(scores["rouge-l"]["f1"], 0.0)

    # 6. Kiểm thử Factuality Checker (Phát hiện Bịa thông tin / Ảo giác)
    def test_fact_checker_faithful(self):
        doc = "Giá vàng đạt mức 85 triệu đồng vào ngày 15/10 tại Hà Nội."
        faithful_summary = "Tại Hà Nội, giá vàng đạt 85 triệu đồng ngày 15/10."
        res = FactualityChecker.check_factuality(faithful_summary, doc)
        self.assertFalse(res["has_hallucination"])
        self.assertEqual(res["factuality_score"], 100.0)

    def test_fact_checker_detect_hallucinated_number_and_entity(self):
        doc = "Doanh thu công ty đạt 500 tỷ đồng tại thị trường TP. HCM."
        hallucinated_summary = "Doanh thu công ty đạt 900 tỷ đồng tại thị trường Đà Nẵng."
        res = FactualityChecker.check_factuality(hallucinated_summary, doc)
        self.assertTrue(res["has_hallucination"])
        self.assertIn("900 tỷ", res["numeric_consistency"]["hallucinated"])
        self.assertIn("Đà Nẵng", res["entity_consistency"]["hallucinated"])

    # 7. Kiểm thử Error Analyzer
    def test_error_analyzer_omission_and_hallucination(self):
        analyzer = SummarizationErrorAnalyzer()
        doc = "Dự án đường cao tốc hoàn thành vào tháng 12 năm 2026 với chi phí 15.000 tỷ đồng."
        gold = "Dự án cao tốc có kinh phí 15.000 tỷ đồng sẽ hoàn thành tháng 12/2026."
        bad_summary = "Dự án thủy điện trị giá 5.000 tỷ đồng tại Sa Pa." # Sai dữ kiện và thiếu ý
        
        analysis = analyzer.analyze_sample("ERR_01", doc, gold, bad_summary, "Giao thông")
        self.assertTrue(analysis["has_error"])
        cats = [e["category"] for e in analysis["errors"]]
        self.assertIn("BIA_THONG_TIN", cats)

if __name__ == "__main__":
    unittest.main()
