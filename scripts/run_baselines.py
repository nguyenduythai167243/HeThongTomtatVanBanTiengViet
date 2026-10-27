"""Script chạy và đánh giá các mô hình đối chuẩn (Baselines):
- Lead-1, Lead-2, Lead-3 Extractive Summarizer
- TF-IDF Extractive Summarizer với hệ số vị trí (Position Prior)
- Đánh giá chỉ số ROUGE-1, ROUGE-2, ROUGE-L trên tập test
- Lưu dự đoán ra file định dạng theo mã mẫu (predictions_baselines.jsonl).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, load_jsonl, save_jsonl, save_json, Timer
from src.baselines.lead_k import LeadKSummarizer
from src.baselines.tfidf_summarizer import TFIDFSummarizer
from src.evaluation.rouge_evaluator import ROUGEEvaluator
from src.evaluation.fact_checker import FactualityChecker

logger = get_logger("RunBaselines")

def main():
    test_file = os.path.join(PROJECT_ROOT, "data", "processed", "test.jsonl")
    if not os.path.exists(test_file):
        test_file = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")

    test_samples = load_jsonl(test_file)
    logger.info(f"Nạp {len(test_samples)} mẫu dữ liệu test từ {test_file}")

    documents = [s["document"] for s in test_samples]
    references = [s["summary"] for s in test_samples]

    evaluator = ROUGEEvaluator()

    models = [
        ("Lead-1", LeadKSummarizer(k=1)),
        ("Lead-2", LeadKSummarizer(k=2)),
        ("Lead-3", LeadKSummarizer(k=3)),
        ("TF-IDF(top-2)", TFIDFSummarizer(num_sentences=2)),
        ("TF-IDF(top-3)", TFIDFSummarizer(num_sentences=3))
    ]

    benchmark_results = {}
    all_predictions = []

    print("\n" + "=" * 78)
    print("      KẾT QUẢ ĐỐI CHUẨN CÁC MÔ HÌNH BASELINE (ROUGE & LATENCY)")
    print("=" * 78)
    print(f"{'Mô hình':<18} | {'ROUGE-1 F1':<12} | {'ROUGE-2 F1':<12} | {'ROUGE-L F1':<12} | {'Latency (ms)':<12}")
    print("-" * 78)

    for name, model in models:
        with Timer() as timer:
            predictions = model.batch_summarize(documents)

        # Đánh giá ROUGE
        rouge_scores = evaluator.evaluate_corpus(predictions, references)
        avg_latency = round(timer.elapsed_ms / len(documents), 2)

        benchmark_results[name] = {
            "rouge_scores": rouge_scores,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": round(timer.elapsed_ms, 2)
        }

        r1_f1 = rouge_scores["rouge-1"]["f1"]
        r2_f1 = rouge_scores["rouge-2"]["f1"]
        rl_f1 = rouge_scores["rouge-l"]["f1"]

        print(f"{name:<18} | {r1_f1:<12.4f} | {r2_f1:<12.4f} | {rl_f1:<12.4f} | {avg_latency:<12.2f}")

        # Ghi nhận kết quả dự đoán
        for sample, pred in zip(test_samples, predictions):
            fact_res = FactualityChecker.check_factuality(pred, sample["document"])
            all_predictions.append({
                "model": name,
                "id": sample["id"],
                "document": sample["document"],
                "reference_summary": sample["summary"],
                "predicted_summary": pred,
                "factuality_score": fact_res["factuality_score"],
                "has_hallucination": fact_res["has_hallucination"]
            })

    print("=" * 78 + "\n")

    # Lưu kết quả và dự đoán
    out_dir = os.path.join(PROJECT_ROOT, "artifacts_and_results")
    os.makedirs(out_dir, exist_ok=True)
    save_json(benchmark_results, os.path.join(out_dir, "baseline_metrics.json"))
    save_jsonl(all_predictions, os.path.join(out_dir, "predictions_baselines.jsonl"))
    logger.info("Hoàn tất lưu baseline metrics và predictions!")

if __name__ == "__main__":
    main()
