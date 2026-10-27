"""Engine xử lý văn bản dài (Long Document Summarization) vượt qua giới hạn độ dài 1024 tokens của Transformer:
1. Truncation: Cắt cụt văn bản tại ngưỡng tối đa (đối chứng baseline).
2. Map-Reduce Chunking: Phân đoạn văn bản có gối đầu (sliding window overlap) -> Tóm tắt từng đoạn (Map) -> Tổng hợp tóm tắt chung (Reduce).
3. Hybrid Extractive-Abstractive: Trích xuất các câu nòng cốt trước khi đưa vào mô hình sinh ngôn ngữ.
"""

from typing import List, Dict, Any, Optional
from src.utils import get_logger, Timer
from src.preprocessor import VietnamesePreprocessor
from src.models.bartpho_summarizer import BARTphoSummarizer

logger = get_logger("ChunkingEngine")

class ChunkingSummarizationEngine:
    """Bộ xử lý văn bản dài tích hợp kỹ thuật Chunking Map-Reduce."""

    def __init__(
        self,
        base_summarizer: BARTphoSummarizer,
        chunk_size_words: int = 350,
        chunk_overlap_words: int = 60,
        strategy: str = "map_reduce"
    ):
        self.summarizer = base_summarizer
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.strategy = strategy

    def split_into_chunks(self, text: str) -> List[str]:
        """Chia văn bản thành các đoạn (chunks) theo ranh giới câu, có độ gối đầu (overlap)."""
        sentences = VietnamesePreprocessor.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sent_words = len(VietnamesePreprocessor.tokenize_syllables(sentence))
            if current_word_count + sent_words > self.chunk_size_words and current_chunk_sentences:
                # Tạo chunk hiện tại
                chunk_str = " ".join(current_chunk_sentences).strip()
                chunks.append(chunk_str)

                # Giữ lại các câu cuối làm overlap cho chunk tiếp theo
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_chunk_sentences):
                    cnt = len(VietnamesePreprocessor.tokenize_syllables(s))
                    if overlap_count + cnt <= self.chunk_overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_count += cnt
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [sentence]
                current_word_count = overlap_count + sent_words
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sent_words

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences).strip())

        return chunks

    def summarize_map_reduce(self, text: str) -> Dict[str, Any]:
        """Tóm tắt văn bản dài theo mô hình Map-Reduce:
        - Giai đoạn Map: Tóm tắt từng chunk độc lập để rút ra ý cốt lõi từng phân đoạn.
        - Giai đoạn Reduce: Gộp các sub-summaries và tóm tắt lại lần cuối để tạo văn bản mạch lạc.
        """
        with Timer() as timer:
            chunks = self.split_into_chunks(text)
            num_chunks = len(chunks)

            if num_chunks <= 1:
                # Văn bản ngắn, không cần chunking
                res = self.summarizer.summarize(text)
                res["num_chunks"] = 1
                res["strategy"] = "direct"
                return res

            logger.info(f"Văn bản dài: Đã phân tách thành {num_chunks} chunks. Bắt đầu giai đoạn Map...")

            # 1. Giai đoạn Map
            sub_summaries = []
            for idx, ch in enumerate(chunks):
                sub_res = self.summarizer.summarize(
                    ch,
                    max_target_length=80,
                    min_target_length=20
                )
                if sub_res.get("summary"):
                    sub_summaries.append(sub_res["summary"])

            # 2. Giai đoạn Reduce
            combined_sub_summaries = " ".join(sub_summaries)
            logger.info(f"Giai đoạn Reduce: Tổng hợp từ {len(sub_summaries)} tóm tắt con...")

            final_res = self.summarizer.summarize(
                combined_sub_summaries,
                max_target_length=150,
                min_target_length=40
            )

        return {
            "summary": final_res.get("summary", ""),
            "status": "success",
            "strategy": "map_reduce_chunking",
            "num_chunks": num_chunks,
            "sub_summaries": sub_summaries,
            "latency_ms": round(timer.elapsed_ms, 2),
            "input_words": len(VietnamesePreprocessor.tokenize_syllables(text)),
            "summary_words": len(VietnamesePreprocessor.tokenize_syllables(final_res.get("summary", "")))
        }

    def summarize_truncation(self, text: str, max_words: int = 500) -> Dict[str, Any]:
        """Chiến lược đối chứng Truncation: Cắt cụt văn bản tại max_words rồi tóm tắt."""
        with Timer() as timer:
            words = text.split()
            truncated_text = " ".join(words[:max_words])
            res = self.summarizer.summarize(truncated_text)

        return {
            "summary": res.get("summary", ""),
            "status": "success",
            "strategy": f"truncation_{max_words}_words",
            "original_words": len(words),
            "truncated_words": len(truncated_text.split()),
            "latency_ms": round(timer.elapsed_ms, 2),
            "info": "Chỉ lấy phần đầu văn bản, bỏ qua toàn bộ thân bài và kết luận phía sau."
        }
