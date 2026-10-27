"""Script chạy đánh giá toàn diện và so sánh đối chuẩn (Comprehensive Evaluation):
- So sánh các mô hình: Lead-1, Lead-3, TF-IDF, BARTpho Seq2Seq, BARTpho + Map-Reduce Chunking
- Đánh giá ROUGE-1/2/L (F1, Precision, Recall)
- Đánh giá Factuality Score (tính nhất quán số liệu và thực thể tên riêng)
- Đo lường chi phí tính toán: Latency (ms), Throughput (từ/giây)
- Xuất toàn bộ dự đoán theo mã mẫu và báo cáo phân tích hồ sơ lỗi (Error Analysis Report).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, load_jsonl, save_jsonl, save_json, Timer
from src.preprocessor import VietnamesePreprocessor
from src.baselines.lead_k import LeadKSummarizer
from src.baselines.tfidf_summarizer import TFIDFSummarizer
from src.models.bartpho_summarizer import BARTphoSummarizer
from src.models.chunking_engine import ChunkingSummarizationEngine
from src.evaluation.rouge_evaluator import ROUGEEvaluator
from src.evaluation.fact_checker import FactualityChecker
from src.evaluation.error_analyzer import SummarizationErrorAnalyzer

logger = get_logger("ComprehensiveEval")

def main():
    test_file = os.path.join(PROJECT_ROOT, "data", "processed", "test.jsonl")
    if not os.path.exists(test_file):
        test_file = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")

    test_samples = load_jsonl(test_file)
    logger.info(f"Nạp {len(test_samples)} mẫu dữ liệu test từ {test_file}")

    documents = [s["document"] for s in test_samples]
    references = [s["summary"] for s in test_samples]

    evaluator = ROUGEEvaluator()
    error_analyzer = SummarizationErrorAnalyzer()

    # Khởi tạo các mô hình
    bartpho = BARTphoSummarizer(lazy_load=True)
    chunking_engine = ChunkingSummarizationEngine(bartpho, chunk_size_words=250, chunk_overlap_words=50)

    models = [
        ("Lead-1", LeadKSummarizer(k=1)),
        ("Lead-3", LeadKSummarizer(k=3)),
        ("TF-IDF(top-3)", TFIDFSummarizer(num_sentences=3)),
        ("BARTpho-Seq2Seq", bartpho),
        ("BARTpho+Chunking", chunking_engine)
    ]

    all_metrics = {}
    test_predictions = []

    print("\n" + "=" * 95)
    print("      BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG TOÀN DIỆN (NLP-20 BENCHMARK)")
    print("=" * 95)
    print(f"{'Mô hình':<20} | {'ROUGE-1 F1':<12} | {'ROUGE-2 F1':<12} | {'ROUGE-L F1':<12} | {'Factuality':<12} | {'Latency':<10}")
    print("-" * 95)

    for name, model in models:
        preds = []
        with Timer() as timer:
            for doc in documents:
                if name == "BARTpho-Seq2Seq":
                    res = model.summarize(doc)
                    preds.append(res.get("summary", ""))
                elif name == "BARTpho+Chunking":
                    res = model.summarize_map_reduce(doc)
                    preds.append(res.get("summary", ""))
                else:
                    preds.append(model.summarize(doc))

        # Đánh giá ROUGE
        rouge = evaluator.evaluate_corpus(preds, references)

        # Đánh giá Factuality
        fact_scores = []
        for p, d in zip(preds, documents):
            f_res = FactualityChecker.check_factuality(p, d)
            fact_scores.append(f_res["factuality_score"])
        avg_factuality = round(sum(fact_scores) / len(fact_scores), 2) if fact_scores else 100.0

        avg_latency = round(timer.elapsed_ms / len(documents), 2)
        total_words = sum(len(VietnamesePreprocessor.tokenize_syllables(d)) for d in documents)
        throughput = round(total_words / (timer.elapsed_ms / 1000.0), 2) if timer.elapsed_ms > 0 else 0.0

        all_metrics[name] = {
            "num_test_samples": len(documents),
            "rouge_1": rouge["rouge-1"],
            "rouge_2": rouge["rouge-2"],
            "rouge_l": rouge["rouge-l"],
            "factuality_score": avg_factuality,
            "avg_latency_ms": avg_latency,
            "throughput_words_per_sec": throughput
        }

        r1 = rouge["rouge-1"]["f1"]
        r2 = rouge["rouge-2"]["f1"]
        rl = rouge["rouge-l"]["f1"]

        print(f"{name:<20} | {r1:<12.4f} | {r2:<12.4f} | {rl:<12.4f} | {avg_factuality:<11.2f}% | {avg_latency:<7.2f} ms")

        # Lưu dự đoán
        for s, p, f_score in zip(test_samples, preds, fact_scores):
            test_predictions.append({
                "model": name,
                "sample_id": s["id"],
                "domain": s.get("domain", "Chung"),
                "document": s["document"],
                "gold_summary": s["summary"],
                "generated_summary": p,
                "factuality_score": f_score
            })

    print("=" * 95 + "\n")

    # Phân tích hồ sơ lỗi cho mô hình chính BARTpho-Seq2Seq
    bartpho_preds = [item["generated_summary"] for item in test_predictions if item["model"] == "BARTpho-Seq2Seq"]
    error_report = error_analyzer.analyze_corpus_errors(test_samples, bartpho_preds)

    # Lưu kết quả
    out_dir = os.path.join(PROJECT_ROOT, "artifacts_and_results")
    os.makedirs(out_dir, exist_ok=True)
    save_json(all_metrics, os.path.join(out_dir, "benchmark_metrics.json"))
    save_jsonl(test_predictions, os.path.join(out_dir, "predictions_test.jsonl"))
    save_json(error_report, os.path.join(out_dir, "error_analysis_report.json"))

    logger.info("Hoàn tất đánh giá toàn diện và lưu trữ dữ liệu vào 'artifacts_and_results/'!")

if __name__ == "__main__":
    main()
