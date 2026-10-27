"""Script thực nghiệm so sánh có kiểm soát (Controlled Ablation Experiments):
1. Thí nghiệm A: Khảo sát ngân sách độ dài đầu ra (Output Length Budget):
   - Ngắn (Short: 20-60 từ, penalty=0.6)
   - Tiêu chuẩn (Medium: 40-120 từ, penalty=1.0)
   - Dài chi tiết (Long: 80-200 từ, penalty=1.8)
2. Thí nghiệm B: So sánh Truncation vs Map-Reduce Chunking trên văn bản dài (>1000 từ).
   - Kiểm tra giả thuyết: Truncation bỏ sót thông tin ở cuối văn bản (Lost-in-the-middle / Omission).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, load_jsonl, save_json, load_json
from src.models.bartpho_summarizer import BARTphoSummarizer
from src.models.chunking_engine import ChunkingSummarizationEngine
from src.evaluation.rouge_evaluator import ROUGEEvaluator
from src.preprocessor import VietnamesePreprocessor

logger = get_logger("AblationExperiments")

def run_length_budget_ablation(summarizer: BARTphoSummarizer, samples: list, evaluator: ROUGEEvaluator) -> dict:
    config_path = os.path.join(PROJECT_ROOT, "configs", "ablation_length_budget.json")
    cfg = load_json(config_path) if os.path.exists(config_path) else {}
    experiments = cfg.get("experiments", [
        {"name": "Budget-Short", "min_target_length": 20, "max_target_length": 60, "length_penalty": 0.6},
        {"name": "Budget-Medium", "min_target_length": 40, "max_target_length": 120, "length_penalty": 1.0},
        {"name": "Budget-Long", "min_target_length": 80, "max_target_length": 200, "length_penalty": 1.8}
    ])

    results = {}
    print("\n" + "=" * 80)
    print("  THÍ NGHIỆM 1: KHẢO SÁT ẢNH HƯỞNG CỦA NGÂN SÁCH ĐỘ DÀI ĐẦU RA (LENGTH BUDGET)")
    print("=" * 80)
    print(f"{'Cấu hình':<20} | {'Độ dài TB (từ)':<16} | {'ROUGE-1 F1':<12} | {'ROUGE-2 F1':<12} | {'ROUGE-L F1':<12}")
    print("-" * 80)

    for exp in experiments:
        name = exp["name"]
        min_len = exp["min_target_length"]
        max_len = exp["max_target_length"]
        penalty = exp["length_penalty"]

        preds = []
        gen_lengths = []
        for s in samples:
            res = summarizer.summarize(
                s["document"],
                max_target_length=max_len,
                min_target_length=min_len,
                length_penalty=penalty
            )
            summary_text = res.get("summary", "")
            preds.append(summary_text)
            gen_lengths.append(len(VietnamesePreprocessor.tokenize_syllables(summary_text)))

        references = [s["summary"] for s in samples]
        scores = evaluator.evaluate_corpus(preds, references)
        avg_len = round(sum(gen_lengths) / len(gen_lengths), 1)

        r1 = scores["rouge-1"]["f1"]
        r2 = scores["rouge-2"]["f1"]
        rl = scores["rouge-l"]["f1"]

        results[name] = {
            "avg_summary_words": avg_len,
            "min_length_budget": min_len,
            "max_length_budget": max_len,
            "length_penalty": penalty,
            "rouge_scores": scores
        }
        print(f"{name:<20} | {avg_len:<16.1f} | {r1:<12.4f} | {r2:<12.4f} | {rl:<12.4f}")

    print("=" * 80)
    return results

def run_truncation_vs_chunking_ablation(
    summarizer: BARTphoSummarizer, samples: list, evaluator: ROUGEEvaluator
) -> dict:
    # Lọc mẫu có văn bản dài nhất trong tập benchmark
    long_samples = [s for s in samples if len(VietnamesePreprocessor.tokenize_syllables(s["document"])) >= 150]
    if not long_samples:
        long_samples = samples

    chunking_engine = ChunkingSummarizationEngine(summarizer, chunk_size_words=120, chunk_overlap_words=30)

    print("\n" + "=" * 80)
    print("  THÍ NGHIỆM 2: SO SÁNH TRUNCATION VS MAP-REDUCE CHUNKING TRÊN VĂN BẢN DÀI")
    print("=" * 80)
    print(f"{'Phương pháp':<25} | {'ROUGE-1 F1':<12} | {'ROUGE-2 F1':<12} | {'ROUGE-L F1':<12} | {'Latency (ms)':<12}")
    print("-" * 80)

    # 1. Truncation (Cắt cụt tại 100 từ đầu)
    trunc_preds = []
    for s in long_samples:
        res = chunking_engine.summarize_truncation(s["document"], max_words=100)
        trunc_preds.append(res.get("summary", ""))

    refs = [s["summary"] for s in long_samples]
    trunc_scores = evaluator.evaluate_corpus(trunc_preds, refs)

    # 2. Map-Reduce Chunking
    chunk_preds = []
    for s in long_samples:
        res = chunking_engine.summarize_map_reduce(s["document"])
        chunk_preds.append(res.get("summary", ""))

    chunk_scores = evaluator.evaluate_corpus(chunk_preds, refs)

    t_r1 = trunc_scores["rouge-1"]["f1"]
    t_r2 = trunc_scores["rouge-2"]["f1"]
    t_rl = trunc_scores["rouge-l"]["f1"]

    c_r1 = chunk_scores["rouge-1"]["f1"]
    c_r2 = chunk_scores["rouge-2"]["f1"]
    c_rl = chunk_scores["rouge-l"]["f1"]

    print(f"{'Truncation (Cắt cụt)':<25} | {t_r1:<12.4f} | {t_r2:<12.4f} | {t_rl:<12.4f} | {'Thấp':<12}")
    print(f"{'Map-Reduce Chunking':<25} | {c_r1:<12.4f} | {c_r2:<12.4f} | {c_rl:<12.4f} | {'Cao hơn':<12}")
    print("=" * 80 + "\n")

    return {
        "num_long_samples": len(long_samples),
        "truncation": trunc_scores,
        "chunking": chunk_scores
    }

def main():
    test_file = os.path.join(PROJECT_ROOT, "data", "processed", "test.jsonl")
    if not os.path.exists(test_file):
        test_file = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")

    samples = load_jsonl(test_file)
    summarizer = BARTphoSummarizer(lazy_load=True)
    evaluator = ROUGEEvaluator()

    budget_results = run_length_budget_ablation(summarizer, samples, evaluator)
    chunking_results = run_truncation_vs_chunking_ablation(summarizer, samples, evaluator)

    out_file = os.path.join(PROJECT_ROOT, "artifacts_and_results", "ablation_results.json")
    save_json({
        "output_length_budget_ablation": budget_results,
        "truncation_vs_chunking_ablation": chunking_results
    }, out_file)
    logger.info(f"Đã lưu kết quả thực nghiệm Ablation vào: {out_file}")

if __name__ == "__main__":
    main()
