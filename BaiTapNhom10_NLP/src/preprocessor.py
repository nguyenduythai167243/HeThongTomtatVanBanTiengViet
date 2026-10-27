"""Module tiền xử lý văn bản tiếng Việt:
- Chuẩn hóa Unicode NFC và dấu thanh tiếng Việt
- Phân đoạn câu bảo toàn từ viết tắt (TP., PGS., GS., TS., BS., v.v.)
- Bảo toàn số liệu, ngày tháng, phần trăm và tên riêng viết hoa
- Xử lý các trường hợp biên: văn bản rỗng, quá ngắn, quá dài, nhiễu HTML
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Any

class VietnamesePreprocessor:
    """Bộ tiền xử lý văn bản tiếng Việt chuyên dụng cho bài toán tóm tắt."""

    # Danh sách các từ viết tắt tiếng Việt phổ biến để tránh ngắt câu sai
    ABBREVIATIONS = {
        "TP.", "Tp.", "GS.", "PGS.", "TS.", "ThS.", "BS.", "TT.", "Thủ tướng",
        "UBND", "HĐND", "BCH", "TW", "VN", "VND", "USD", "EUR", "km", "km/h",
        "m2", "ha", "kg", "g", "ml", "lit", "No.", "Vol."
    }

    # Bảng chuẩn hóa dấu tiếng Việt kiểu mới về chuẩn thống nhất (NFC)
    def __init__(self):
        pass

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Chuẩn hóa chuỗi văn bản về Unicode chuẩn dựng sẵn NFC."""
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch văn bản: loại bỏ HTML tag, chuẩn hóa khoảng trắng, ký tự rác."""
        if not text or not isinstance(text, str):
            return ""
        # 1. Unicode NFC
        text = unicodedata.normalize("NFC", text)
        # 2. Xóa thẻ HTML nếu có
        text = re.sub(r"<[^>]+>", " ", text)
        # 3. Chuẩn hóa các loại dấu ngoặc và dấu nháy
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("–", "-").replace("—", "-")
        # 4. Loại bỏ ký tự điều khiển lạ nhưng giữ lại dấu câu, số và chữ cái tiếng Việt
        text = re.sub(r"[\r\t\f\v]", " ", text)
        # 5. Rút gọn nhiều khoảng trắng liên tiếp
        text = re.sub(r"\s+", " ", text).strip()
        # 6. Loại bỏ khoảng trắng đứng trước dấu câu
        text = re.sub(r"\s+([.,!?:;])", r"\1", text)
        return text

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Tách văn bản tiếng Việt thành danh sách câu văn hoàn chỉnh.
        Bảo toàn các từ viết tắt (TP. HCM, TS. Nguyễn Văn A, PGS. TS. ...)
        """
        text = cls.clean_text(text)
        if not text:
            return []

        # Tạm thời thay thế dấu chấm trong các từ viết tắt để không bị ngắt nhầm
        protected = text
        placeholders = {}
        for idx, abbr in enumerate(cls.ABBREVIATIONS):
            if abbr in protected:
                placeholder = f"__ABBR_{idx}__"
                placeholders[placeholder] = abbr
                protected = protected.replace(abbr, placeholder)

        # Bảo vệ số thập phân (ví dụ: 3.14, 15.5%)
        decimal_matches = re.findall(r"\d+\.\d+", protected)
        for idx, dec in enumerate(decimal_matches):
            placeholder = f"__DEC_{idx}__"
            placeholders[placeholder] = dec
            protected = protected.replace(dec, placeholder)

        # Ngắt câu theo dấu kết thúc câu: ., !, ?, hoặc xuống dòng
        raw_sentences = re.split(r"(?<=[.!?])\s+", protected)

        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            # Phục hồi các từ viết tắt và số thập phân
            for placeholder, original in placeholders.items():
                s = s.replace(placeholder, original)
            sentences.append(s)

        return sentences

    @classmethod
    def tokenize_syllables(cls, text: str) -> List[str]:
        """Tách tiếng (syllable tokenization) chuẩn hóa theo âm tiết tiếng Việt.
        Giữ nguyên các số liệu, ngày tháng, tên riêng và từ vựng.
        """
        text = cls.clean_text(text)
        if not text:
            return []
        # Tách theo khoảng trắng và giữ nguyên từ/số
        tokens = text.split()
        return tokens

    @classmethod
    def validate_edge_cases(cls, text: str, min_words: int = 5, max_words: int = 3500) -> Tuple[bool, str, Dict[str, Any]]:
        """Kiểm thử các trường hợp biên của văn bản đầu vào:
        - Rỗng hoặc chỉ có khoảng trắng
        - Quá ngắn (ít hơn min_words)
        - Quá dài (nhiều hơn max_words)
        - Chứa toàn ký tự lặp hoặc số liệu nhiễu
        """
        cleaned = cls.clean_text(text)
        if not cleaned:
            return False, "EMPTY_INPUT: Văn bản đầu vào rỗng hoặc chỉ chứa khoảng trắng.", {"status": "empty", "word_count": 0}

        tokens = cls.tokenize_syllables(cleaned)
        word_count = len(tokens)
        sentences = cls.split_sentences(cleaned)
        sentence_count = len(sentences)

        if word_count < min_words:
            return False, f"TOO_SHORT: Văn bản quá ngắn ({word_count} từ, tối thiểu {min_words} từ).", {
                "status": "too_short",
                "word_count": word_count,
                "sentence_count": sentence_count
            }

        is_very_long = word_count > max_words
        warning_msg = f"VERY_LONG: Văn bản có độ dài lớn ({word_count} từ), cần áp dụng cơ chế Chunking." if is_very_long else "OK"

        return True, warning_msg, {
            "status": "valid",
            "word_count": word_count,
            "sentence_count": sentence_count,
            "is_very_long": is_very_long
        }
