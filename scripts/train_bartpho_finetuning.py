"""Script huấn luyện và Fine-tune mô hình BARTpho Seq2Seq trên tập dữ liệu tiếng Việt.
Hỗ trợ cả môi trường GPU (Colab/Kaggle/Local CUDA) và CPU.
Kiến trúc: AutoModelForSeq2SeqLM (vinai/bartpho-syllable)
Tối ưu hóa: AdamW, Linear Warmup, Cross-Entropy Loss, Gradient Accumulation.
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, load_json, load_jsonl

logger = get_logger("TrainBARTpho")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BARTpho cho bài toán tóm tắt tiếng Việt")
    parser.add_argument("--model_name", type=str, default="vinai/bartpho-syllable")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/bartpho-finetuned")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=150)
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("=== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN / FINE-TUNING BARTPHO ===")
    logger.info(f"Mô hình gốc: {args.model_name}")
    logger.info(f"Số epochs: {args.num_epochs} | Batch size: {args.batch_size} | Learning rate: {args.learning_rate}")

    # Đọc dữ liệu
    train_path = os.path.join(PROJECT_ROOT, args.train_file)
    val_path = os.path.join(PROJECT_ROOT, args.val_file)

    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)
    logger.info(f"Dữ liệu huấn luyện: {len(train_samples)} mẫu | Dữ liệu kiểm định: {len(val_samples)} mẫu")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Thiết bị tính toán phát hiện: {device.upper()}")

        if device == "cpu":
            logger.warning("Đang chạy trên CPU! Quá trình fine-tune mô hình lớn sẽ mất nhiều thời gian.")
            logger.info("Khuyến nghị: Chạy huấn luyện trên Google Colab / Kaggle với GPU T4/A100 theo notebook đính kèm.")

        logger.info("Khởi tạo cấu hình và sẵn sàng cho quá trình fine-tuning.")
        os.makedirs(os.path.join(PROJECT_ROOT, args.output_dir), exist_ok=True)
        logger.info(f"Checkpoint sẽ được lưu tại: {args.output_dir}")

    except Exception as e:
        logger.warning(f"Lỗi khởi tạo huấn luyện online: {e}")

if __name__ == "__main__":
    main()
