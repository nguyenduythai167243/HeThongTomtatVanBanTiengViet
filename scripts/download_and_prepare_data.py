"""Script khởi tạo và chuẩn bị bộ dữ liệu tóm tắt văn bản tiếng Việt chuẩn theo schema.json.
- TẬP HỢP 5 LĨNH VỰC TOÀN DIỆN:
  1. Kinh tế - Tài chính (240 mẫu từ bộ chuẩn VietNews)
  2. Khoa học - Công nghệ (15 mẫu thực tế)
  3. Y tế - Sức khỏe (15 mẫu thực tế)
  4. Giáo dục - Đào tạo (15 mẫu thực tế)
  5. Chính trị - Xã hội (15 mẫu thực tế)
  Tổng cộng: 300 mẫu chuẩn, đa dạng ngữ cảnh và trường hợp biên (văn bản dài >700 từ, mật độ số liệu cao).
- PHÂN CHIA PHÂN TẦNG (STRATIFIED SPLIT): Train (70%), Val (15%), Test (15%) theo từng domain để chống rò rỉ dữ liệu (leakage).
"""

import os
import sys
import json
import urllib.request
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, save_jsonl, save_json, load_jsonl
from src.preprocessor import VietnamesePreprocessor
from src.data_loader import remove_near_duplicates, split_dataset_by_domain

logger = get_logger("PrepareData")

VIETNEWS_URL = "https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum/resolve/main/processed/test_desegmented.jsonl"
MULTIDOMAIN_CACHE = os.path.join(PROJECT_ROOT, "data", "raw_domains", "multidomain_samples.jsonl")

KEYWORDS_ECONOMY = [
    "ngân hàng", "lãi suất", "chứng khoán", "cổ phiếu", "doanh nghiệp", "kinh tế",
    "xuất khẩu", "nhập khẩu", "tài chính", "bất động sản", "doanh thu", "lợi nhuận",
    "lạm phát", "thương mại", "giá vàng", "tỷ giá", "gdp", "thị trường", "đầu tư",
    "fdi", "giá xăng", "giá dầu", "tiền tệ", "trái phiếu", "kinh doanh", "vốn"
]

EXCLUDE_TOPICS = [
    "công an", "cảnh sát", "nghi phạm", "tạm giữ", "tòa án", "phiên toà", "phiên tòa",
    "truy nã", "tín dụng đen", "vụ án", "giết", "chết", "tai nạn", "trộm", "cướp",
    "ma túy", "khởi tố", "điều tra", "bắt quả tang", "bị phạt", "tạt axit", "bệnh viện",
    "bệnh nhân", "bác sĩ", "tử vong", "ly hôn", "thủ dâm", "hiếp dâm", "đánh ghen",
    "cháy nhà", "hỏa hoạn", "lừa đảo", "cờ bạc", "đánh bạc", "xâm hại"
]

def fetch_vietnews_economic_samples(target_count: int = 240) -> list:
    """Tải và lọc dữ liệu VietNews chuẩn cho miền Kinh tế - Tài chính."""
    # Nếu đã có dữ liệu mẫu trước đó trong sample_benchmark.jsonl, tận dụng để tăng tốc
    benchmark_path = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")
    if os.path.exists(benchmark_path):
        existing = load_jsonl(benchmark_path)
        eco_existing = [x for x in existing if x.get("domain") == "Kinh tế - Tài chính"]
        if len(eco_existing) >= target_count:
            logger.info(f"Đã nạp {target_count} mẫu Kinh tế - Tài chính từ bộ nhớ đệm cục bộ.")
            return eco_existing[:target_count]

    logger.info(f"Đang tải dữ liệu từ kho VietNews: {VIETNEWS_URL}")
    req = urllib.request.Request(VIETNEWS_URL, headers={"User-Agent": "Mozilla/5.0"})
    collected = []
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for idx, raw_line in enumerate(resp):
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line.decode("utf-8"))
                    title = item.get("title", "").strip()
                    abstract = item.get("abstract", "").strip()
                    article = item.get("article", "").strip()

                    doc_words = len(article.split())
                    sum_words = len(abstract.split())
                    if doc_words < 80 or sum_words < 12:
                        continue

                    combined = (title + " " + abstract).lower()
                    if any(bad in combined for bad in EXCLUDE_TOPICS):
                        continue
                    if not any(kw in combined for kw in KEYWORDS_ECONOMY):
                        continue

                    is_long = doc_words >= 600
                    has_heavy_numbers = len([w for w in abstract.split() if any(c.isdigit() for c in w)]) >= 3

                    edge_type = "normal"
                    if is_long:
                        edge_type = "very_long"
                    elif has_heavy_numbers:
                        edge_type = "heavy_numbers"

                    sample = {
                        "id": f"VN_ECO_{len(collected)+1:03d}",
                        "title": title,
                        "document": article,
                        "summary": abstract,
                        "domain": "Kinh tế - Tài chính",
                        "source": "VietNews (BARTpho Benchmark Corpus)",
                        "metadata": {
                            "is_edge_case": is_long or has_heavy_numbers,
                            "edge_case_type": edge_type
                        }
                    }
                    collected.append(sample)
                    if len(collected) >= target_count:
                        break
                except Exception:
                    continue
                if idx > 9000:
                    break
        return collected
    except Exception as e:
        logger.error(f"Lỗi tải VietNews: {e}")
        return []

def load_or_fetch_multidomain_samples() -> list:
    """Nạp 60 mẫu đại diện cho 4 lĩnh vực: Khoa học - Công nghệ, Y tế - Sức khỏe, Giáo dục, Chính trị - Xã hội."""
    if os.path.exists(MULTIDOMAIN_CACHE):
        samples = load_jsonl(MULTIDOMAIN_CACHE)
        logger.info(f"Đã nạp {len(samples)} mẫu từ 4 lĩnh vực bổ sung: {MULTIDOMAIN_CACHE}")
        return samples

    rss_feeds = {
        'Khoa học - Công nghệ': 'https://vnexpress.net/rss/so-hoa.rss',
        'Y tế - Sức khỏe': 'https://vnexpress.net/rss/suc-khoe.rss',
        'Giáo dục - Đào tạo': 'https://vnexpress.net/rss/giao-duc.rss',
        'Chính trị - Xã hội': 'https://vnexpress.net/rss/thoi-su.rss'
    }
    all_samples = []
    for domain, url in rss_feeds.items():
        logger.info(f"Đang thu thập mẫu cho lĩnh vực: {domain}...")
        try:
            import xml.etree.ElementTree as ET
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(urllib.request.urlopen(req, timeout=8).read())
            items = root.findall('.//item')
            collected = []
            prefix = {
                'Khoa học - Công nghệ': 'TECH',
                'Y tế - Sức khỏe': 'HEALTH',
                'Giáo dục - Đào tạo': 'EDU',
                'Chính trị - Xã hội': 'SOC'
            }[domain]

            for it in items:
                try:
                    title = it.find('title').text.strip()
                    link = it.find('link').text.strip()
                    desc = it.find('description').text or ''
                    summary = re.sub(r'<[^>]+>', '', desc).strip()
                    if len(summary.split()) < 10:
                        continue

                    art_req = urllib.request.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
                    html = urllib.request.urlopen(art_req, timeout=6).read().decode('utf-8', errors='ignore')
                    paras = re.findall(r'<p class=.Normal.>(.*?)</p>', html, re.DOTALL)
                    clean_paras = [re.sub(r'<[^>]+>', '', p).strip() for p in paras]
                    doc_body = ' '.join([p for p in clean_paras if p and not p.startswith('Ảnh:')])
                    if len(doc_body.split()) < 80:
                        continue

                    sample = {
                        'id': f'VN_{prefix}_{len(collected)+1:03d}',
                        'title': title,
                        'summary': summary,
                        'document': doc_body,
                        'domain': domain,
                        'source': 'VnExpress',
                        'metadata': {
                            'is_edge_case': len(doc_body.split()) > 700,
                            'edge_case_type': 'long_document' if len(doc_body.split()) > 700 else 'normal'
                        }
                    }
                    collected.append(sample)
                    if len(collected) >= 15:
                        break
                except Exception:
                    continue
            logger.info(f"Đã thu thập {len(collected)} mẫu cho lĩnh vực: {domain}")
            all_samples.extend(collected)
        except Exception as e:
            logger.error(f"Lỗi thu thập lĩnh vực {domain}: {e}")

    os.makedirs(os.path.dirname(MULTIDOMAIN_CACHE), exist_ok=True)
    save_jsonl(all_samples, MULTIDOMAIN_CACHE)
    return all_samples

def main():
    logger.info("=== BẮT ĐẦU CHUẨN BỊ BỘ DỮ LIỆU ĐA LĨNH VỰC (5 DOMAINS - 300 MẪU CHUẨN) ===")

    # 1. Thu thập 240 mẫu Kinh tế - Tài chính
    eco_samples = fetch_vietnews_economic_samples(target_count=240)
    logger.info(f"1. Kinh tế - Tài chính: {len(eco_samples)} mẫu.")

    # 2. Thu thập 60 mẫu từ 4 lĩnh vực bổ sung (15 mẫu mỗi lĩnh vực)
    multi_samples = load_or_fetch_multidomain_samples()
    logger.info(f"2. Bốn lĩnh vực bổ sung: {len(multi_samples)} mẫu.")

    total_samples = eco_samples + multi_samples
    logger.info(f"Tổng số mẫu thô trước khi deduplication: {len(total_samples)} mẫu.")

    # 3. Tiền xử lý, chuẩn hóa tiếng Việt và bổ sung chỉ số định lượng
    enriched = []
    for s in total_samples:
        clean_doc = VietnamesePreprocessor.clean_text(s["document"])
        clean_sum = VietnamesePreprocessor.clean_text(s["summary"])
        doc_words = len(VietnamesePreprocessor.tokenize_syllables(clean_doc))
        sum_words = len(VietnamesePreprocessor.tokenize_syllables(clean_sum))
        doc_sents = len(VietnamesePreprocessor.split_sentences(clean_doc))

        scopy = dict(s)
        scopy["document"] = clean_doc
        scopy["summary"] = clean_sum
        scopy["word_count_doc"] = doc_words
        scopy["word_count_summary"] = sum_words
        scopy["sentence_count_doc"] = doc_sents
        scopy["compression_ratio"] = round(sum_words / doc_words, 4) if doc_words > 0 else 0.0
        enriched.append(scopy)

    # 4. Lọc trùng lặp Jaccard (Deduplication)
    unique_samples = remove_near_duplicates(enriched, similarity_threshold=0.85)

    # 5. Lưu toàn bộ 300 mẫu benchmark đa lĩnh vực
    benchmark_path = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")
    save_jsonl(unique_samples, benchmark_path)
    logger.info(f"Đã lưu toàn bộ {len(unique_samples)} mẫu vào: {benchmark_path}")

    # 6. Phân chia Train/Val/Test theo phân tầng Domain chống rò rỉ dữ liệu
    train_data, val_data, test_data = split_dataset_by_domain(
        unique_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    save_jsonl(train_data, os.path.join(processed_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(processed_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(processed_dir, "test.jsonl"))

    logger.info(f"Phân chia hoàn tất: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")
    logger.info("=== HOÀN TẤT CHUẨN BỊ BỘ DỮ LIỆU ĐA LĨNH VỰC ===")

if __name__ == "__main__":
    main()
