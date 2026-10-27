"""Module chứa các hàm tiện ích chung: logging, đọc/ghi JSONL, đo thời gian latency, quản lý tệp tin."""

import os
import json
import time
import logging
from typing import List, Dict, Any, Generator

def get_logger(name: str = "NLP-20") -> logging.Logger:
    """Tạo logger chuẩn cho hệ thống với format rõ ràng."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class Timer:
    """Class hỗ trợ đo lường thời gian thực thi (latency / throughput)."""
    def __init__(self):
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

def load_json(filepath: str) -> Dict[str, Any]:
    """Đọc tệp tin JSON an toàn."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Lưu tệp tin JSON với thụt dòng dễ đọc."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu từ file định dạng JSON Lines."""
    samples = []
    if not os.path.exists(filepath):
        return samples
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                get_logger().warning(f"Lỗi cú pháp JSON tại dòng {line_num} trong file {filepath}: {e}")
    return samples

def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Ghi danh sách bản ghi ra file định dạng JSON Lines."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
