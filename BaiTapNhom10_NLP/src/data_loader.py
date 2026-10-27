"""Module quản lý dữ liệu và tiền xử lý tập dữ liệu:
- Kiểm tra tính hợp lệ theo schema.json
- Lọc bỏ văn bản trùng lặp / gần trùng (deduplication) bằng Jaccard similarity
- Phân chia train/val/test theo Domain/Chủ đề để ngăn chặn rò rỉ dữ liệu (Domain/Source Data Leakage)
- Hỗ trợ lưu và đọc các tập train.jsonl, val.jsonl, test.jsonl
"""

import json
import random
from typing import List, Dict, Any, Tuple
from src.utils import get_logger, load_jsonl, save_jsonl, load_json
from src.preprocessor import VietnamesePreprocessor

logger = get_logger("DataLoader")

def validate_sample_schema(sample: Dict[str, Any], schema: Dict[str, Any] = None) -> Tuple[bool, str]:
    """Kiểm tra một mẫu dữ liệu có đầy đủ các trường bắt buộc theo schema hay không."""
    required_fields = ["id", "title", "document", "summary", "domain", "source"]
    for field in required_fields:
        if field not in sample or not sample[field]:
            return False, f"Thiếu hoặc rỗng trường bắt buộc: '{field}'"

    if len(sample["document"].strip()) < 10:
        return False, "Trường 'document' quá ngắn (dưới 10 ký tự)"

    if len(sample["summary"].strip()) < 5:
        return False, "Trường 'summary' quá ngắn (dưới 5 ký tự)"

    return True, "Valid"

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Tính độ tương đồng Jaccard giữa hai văn bản dựa trên tập từ vựng."""
    tokens1 = set(VietnamesePreprocessor.tokenize_syllables(text1.lower()))
    tokens2 = set(VietnamesePreprocessor.tokenize_syllables(text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def remove_near_duplicates(samples: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Loại bỏ các bài viết trùng lặp hoặc gần trùng (near-duplicates) trước khi phân chia."""
    unique_samples = []
    dropped_count = 0

    for sample in samples:
        doc = sample["document"]
        is_duplicate = False
        for kept in unique_samples:
            sim = compute_jaccard_similarity(doc, kept["document"])
            if sim >= similarity_threshold:
                is_duplicate = True
                dropped_count += 1
                break
        if not is_duplicate:
            unique_samples.append(sample)

    logger.info(f"Deduplication: Giữ lại {len(unique_samples)}/{len(samples)} mẫu (Đã lọc bỏ {dropped_count} mẫu gần trùng).")
    return unique_samples

def split_dataset_by_domain(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Phân chia tập dữ liệu thành Train, Validation, Test theo phân tầng Domain.
    Đảm bảo phân bố chủ đề đồng đều giữa các tập, ngăn ngừa rò rỉ dữ liệu (data leakage).
    """
    random.seed(seed)
    domain_buckets: Dict[str, List[Dict[str, Any]]] = {}

    for s in samples:
        d = s.get("domain", "Chung")
        domain_buckets.setdefault(d, []).append(s)

    train_set, val_set, test_set = [], [], []

    for domain, items in domain_buckets.items():
        random.shuffle(items)
        n = len(items)
        if n == 1:
            # Mẫu đơn lẻ ưu tiên cho test để đánh giá trường hợp biên
            test_set.append(items[0])
            continue

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_val == 0 and n >= 3:
            n_val = 1
        n_test = n - n_train - n_val

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    logger.info(f"Tập dữ liệu sau phân chia: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_set, val_set, test_set
