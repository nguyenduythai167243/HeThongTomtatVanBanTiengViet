"""Module kiểm tra tính chân thực và phát hiện ảo giác/bịa thông tin (Factuality & Hallucination Checker).
Giải quyết vấn đề cốt lõi của đề tài: "Bịa thông tin (Hallucination) - Sai lệch số liệu và thực thể tên riêng".
ROUGE chỉ đo độ trùng khớp n-gram từ vựng, không phản ánh tính chính xác của dữ kiện.
Hệ thống này kiểm tra sự nhất quán giữa bản tóm tắt và văn bản gốc:
1. Tính nhất quán số liệu (Numeric consistency): Số, ngày tháng, tỷ lệ %, tiền tệ.
2. Tính nhất quán thực thể tên riêng (Named entity consistency): Tên người, địa danh, tổ chức.
"""

import re
from typing import List, Dict, Any, Set, Tuple

class FactualityChecker:
    """Bộ kiểm tra dữ kiện và cảnh báo ảo giác (Hallucination Alert)."""

    # Biểu thức chính quy phát hiện số liệu: số nguyên, số thực, tỷ lệ %, ngày tháng năm, tiền tệ
    NUMBER_PATTERN = re.compile(
        r"\b\d+(?:[.,]\d+)*(?:\s*(?:%|triệu|tỷ|nghìn|USD|VND|đồng|km|ha|kg|m2|năm|tháng|ngày|giờ))?\b",
        re.IGNORECASE
    )

    # Biểu thức phát hiện cụm từ viết hoa (Thực thể tên riêng / Proper Nouns)
    PROPER_NOUN_PATTERN = re.compile(
        r"\b[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+"
        r"(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+)*"
    )

    @classmethod
    def extract_numbers(cls, text: str) -> List[str]:
        """Trích xuất tất cả các số liệu trong văn bản."""
        matches = cls.NUMBER_PATTERN.findall(text)
        return [m.strip().lower() for m in matches if m.strip()]

    @classmethod
    def extract_named_entities(cls, text: str) -> List[str]:
        """Trích xuất các cụm danh từ riêng viết hoa (tên người, tổ chức, địa danh)."""
        matches = cls.PROPER_NOUN_PATTERN.findall(text)
        # Bỏ qua từ viết hoa chỉ vì đứng đầu câu
        clean_entities = []
        for m in matches:
            if len(m.split()) >= 1 and len(m) > 2:
                clean_entities.append(m.strip())
        return list(set(clean_entities))

    @classmethod
    def check_factuality(cls, summary: str, document: str) -> Dict[str, Any]:
        """Kiểm tra xem tóm tắt có sinh ra số liệu hoặc tên riêng không có trong bài gốc hay không.
        Trả về điểm Factuality [0.0, 1.0] và danh sách các lỗi bịa thông tin phát hiện được.
        """
        doc_lower = document.lower()
        summary_numbers = cls.extract_numbers(summary)
        summary_entities = cls.extract_named_entities(summary)

        # 1. Kiểm tra số liệu
        hallucinated_numbers = []
        for num in summary_numbers:
            # Tách riêng phần số cốt lõi để kiểm tra
            digits = re.findall(r"\d+", num)
            for d in digits:
                if d not in doc_lower:
                    hallucinated_numbers.append(num)
                    break

        # 2. Kiểm tra thực thể tên riêng
        hallucinated_entities = []
        for ent in summary_entities:
            if ent.lower() not in doc_lower:
                hallucinated_entities.append(ent)

        total_numbers = len(summary_numbers)
        valid_numbers = total_numbers - len(hallucinated_numbers)
        num_score = valid_numbers / total_numbers if total_numbers > 0 else 1.0

        total_entities = len(summary_entities)
        valid_entities = total_entities - len(hallucinated_entities)
        ent_score = valid_entities / total_entities if total_entities > 0 else 1.0

        overall_factuality = round((num_score * 0.5 + ent_score * 0.5) * 100, 2)
        has_hallucination = (len(hallucinated_numbers) > 0) or (len(hallucinated_entities) > 0)

        return {
            "factuality_score": overall_factuality,
            "has_hallucination": has_hallucination,
            "numeric_consistency": {
                "score": round(num_score * 100, 2),
                "total_in_summary": total_numbers,
                "hallucinated": hallucinated_numbers
            },
            "entity_consistency": {
                "score": round(ent_score * 100, 2),
                "total_in_summary": total_entities,
                "hallucinated": hallucinated_entities
            },
            "verdict": "FAITHFUL" if not has_hallucination else "HALLUCINATED_ALERT"
        }
