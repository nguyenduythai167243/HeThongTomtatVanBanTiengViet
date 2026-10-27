"""Script khảo sát dữ liệu (Exploratory Data Analysis - EDA):
- Thống kê phân bố độ dài văn bản gốc và tóm tắt (từ, câu, ký tự)
- Tính tỷ lệ nén (Compression Ratio)
- Phân tích độ phong phú từ vựng (Vocabulary Size, Type-Token Ratio TTR)
- Thống kê tỷ lệ phân bố theo Domain và các trường hợp biên (Edge Cases)
- Xuất báo cáo ra định dạng JSON và bảng trực quan.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, load_jsonl, save_json
from src.preprocessor import VietnamesePreprocessor

logger = get_logger("EDA")

def run_eda(filepath: str) -> dict:
    samples = load_jsonl(filepath)
    if not samples:
        logger.error(f"Tệp dữ liệu rỗng hoặc không tồn tại: {filepath}")
        return {}

    doc_lengths = []
    summary_lengths = []
    compression_ratios = []
    doc_sentence_counts = []
    all_doc_tokens = []
    domain_counts = {}
    edge_case_counts = {}

    for s in samples:
        doc = s["document"]
        summary = s["summary"]
        domain = s.get("domain", "Không rõ")
        edge_type = s.get("metadata", {}).get("edge_case_type", "normal")

        doc_tokens = VietnamesePreprocessor.tokenize_syllables(doc.lower())
        sum_tokens = VietnamesePreprocessor.tokenize_syllables(summary.lower())
        sents = VietnamesePreprocessor.split_sentences(doc)

        doc_lengths.append(len(doc_tokens))
        summary_lengths.append(len(sum_tokens))
        doc_sentence_counts.append(len(sents))
        all_doc_tokens.extend(doc_tokens)

        ratio = len(sum_tokens) / len(doc_tokens) if len(doc_tokens) > 0 else 0
        compression_ratios.append(ratio)

        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        edge_case_counts[edge_type] = edge_case_counts.get(edge_type, 0) + 1

    vocab_unique = set(all_doc_tokens)
    ttr = len(vocab_unique) / len(all_doc_tokens) if all_doc_tokens else 0.0

    stats = {
        "dataset_filepath": filepath,
        "total_samples": len(samples),
        "document_stats": {
            "mean_words": round(float(np.mean(doc_lengths)), 2),
            "median_words": round(float(np.median(doc_lengths)), 2),
            "min_words": int(np.min(doc_lengths)),
            "max_words": int(np.max(doc_lengths)),
            "std_words": round(float(np.std(doc_lengths)), 2),
            "mean_sentences": round(float(np.mean(doc_sentence_counts)), 2)
        },
        "summary_stats": {
            "mean_words": round(float(np.mean(summary_lengths)), 2),
            "median_words": round(float(np.median(summary_lengths)), 2),
            "min_words": int(np.min(summary_lengths)),
            "max_words": int(np.max(summary_lengths)),
            "std_words": round(float(np.std(summary_lengths)), 2)
        },
        "compression_ratio_stats": {
            "mean_ratio": round(float(np.mean(compression_ratios)), 4),
            "median_ratio": round(float(np.median(compression_ratios)), 4),
            "min_ratio": round(float(np.min(compression_ratios)), 4),
            "max_ratio": round(float(np.max(compression_ratios)), 4)
        },
        "vocabulary_stats": {
            "total_tokens": len(all_doc_tokens),
            "unique_vocab_size": len(vocab_unique),
            "type_token_ratio_ttr": round(ttr, 4)
        },
        "domain_distribution": domain_counts,
        "edge_case_distribution": edge_case_counts
    }

    # In kết quả dạng bảng đẹp mắt
    print("\n" + "=" * 65)
    print("      BÁO CÁO KHẢO SÁT THỐNG KÊ DỮ LIỆU (EDA REPORT)")
    print("=" * 65)
    print(f"Tổng số mẫu khảo sát: {stats['total_samples']}")
    print("-" * 65)
    print("1. Độ dài Văn bản gốc (Document):")
    print(f"   - Trung bình: {stats['document_stats']['mean_words']} từ (Độ lệch chuẩn: {stats['document_stats']['std_words']})")
    print(f"   - Trung vị:   {stats['document_stats']['median_words']} từ | [Min: {stats['document_stats']['min_words']}, Max: {stats['document_stats']['max_words']}]")
    print(f"   - Số câu TB:  {stats['document_stats']['mean_sentences']} câu/bài")
    print("-" * 65)
    print("2. Độ dài Tóm tắt vàng (Summary):")
    print(f"   - Trung bình: {stats['summary_stats']['mean_words']} từ (Độ lệch chuẩn: {stats['summary_stats']['std_words']})")
    print(f"   - Trung vị:   {stats['summary_stats']['median_words']} từ | [Min: {stats['summary_stats']['min_words']}, Max: {stats['summary_stats']['max_words']}]")
    print("-" * 65)
    print("3. Tỷ lệ nén (Compression Ratio = Summary / Doc):")
    print(f"   - Tỷ lệ nén trung bình: {stats['compression_ratio_stats']['mean_ratio'] * 100:.2f}%")
    print(f"   - Tỷ lệ nén Min - Max:  {stats['compression_ratio_stats']['min_ratio'] * 100:.2f}% - {stats['compression_ratio_stats']['max_ratio'] * 100:.2f}%")
    print("-" * 65)
    print("4. Thống kê Từ vựng:")
    print(f"   - Tổng tokens: {stats['vocabulary_stats']['total_tokens']}")
    print(f"   - Từ vựng duy nhất (Vocab size): {stats['vocabulary_stats']['unique_vocab_size']}")
    print(f"   - Tỷ lệ phong phú từ vựng (TTR): {stats['vocabulary_stats']['type_token_ratio_ttr']}")
    print("-" * 65)
    print("5. Phân bố theo Lĩnh vực (Domains):")
    for d, c in domain_counts.items():
        print(f"   - {d:<25}: {c} mẫu ({c/stats['total_samples']*100:.1f}%)")
    print("=" * 65 + "\n")

    out_path = os.path.join(PROJECT_ROOT, "artifacts_and_results", "eda_statistics.json")
    save_json(stats, out_path)
    logger.info(f"Đã lưu kết quả khảo sát EDA vào: {out_path}")
    return stats

if __name__ == "__main__":
    benchmark_file = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")
    run_eda(benchmark_file)
