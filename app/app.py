"""Web Application Demo: Hệ thống tóm tắt văn bản tiếng Việt sử dụng Transformer.
Nhóm 10 - Khoa CNTT - UNETI (Đề tài NLP-20).
Cung cấp REST API và giao diện Web trực quan:
- Tóm tắt văn bản đơn lẻ (Lead-k, TF-IDF, BARTpho, Chunking Engine)
- So sánh song song (Side-by-side comparison) giữa tất cả các mô hình
- Fact-Checking visualizer: phát hiện và cảnh báo bịa thông tin (hallucination)
- Thống kê thời gian xử lý (latency ms) và số lượng từ nén.
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, load_jsonl, load_json, Timer
from src.preprocessor import VietnamesePreprocessor
from src.baselines.lead_k import LeadKSummarizer
from src.baselines.tfidf_summarizer import TFIDFSummarizer
from src.models.bartpho_summarizer import BARTphoSummarizer
from src.models.chunking_engine import ChunkingSummarizationEngine
from src.evaluation.fact_checker import FactualityChecker
from src.evaluation.rouge_evaluator import ROUGEEvaluator

logger = get_logger("WebDemoApp")

# Khởi tạo các mô hình sẵn sàng phục vụ inference
bartpho_model = BARTphoSummarizer(lazy_load=True)
chunking_model = ChunkingSummarizationEngine(bartpho_model, chunk_size_words=250, chunk_overlap_words=50)
lead3_model = LeadKSummarizer(k=3)
tfidf_model = TFIDFSummarizer(num_sentences=3)
evaluator = ROUGEEvaluator()

class SummarizationRequestHandler(SimpleHTTPRequestHandler):
    """Handler xử lý các yêu cầu HTTP GET và POST cho Web Demo."""

    def __init__(self, *args, **kwargs):
        # Thiết lập thư mục tĩnh static
        static_dir = os.path.join(PROJECT_ROOT, "app", "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        if path == "/api/samples":
            self.handle_get_samples()
        elif path == "/api/metrics":
            self.handle_get_metrics()
        else:
            super().do_GET()

    def do_POST(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req_json = {}

        if path == "/api/summarize":
            self.handle_summarize(req_json)
        elif path == "/api/compare":
            self.handle_compare_all(req_json)
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data: dict, status_code: int = 200):
        response_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_get_samples(self):
        sample_path = os.path.join(PROJECT_ROOT, "data", "sample_benchmark.jsonl")
        samples = load_jsonl(sample_path)
        self.send_json({"status": "success", "samples": samples})

    def handle_get_metrics(self):
        metric_path = os.path.join(PROJECT_ROOT, "artifacts_and_results", "benchmark_metrics.json")
        metrics = load_json(metric_path) if os.path.exists(metric_path) else {}
        self.send_json({"status": "success", "metrics": metrics})

    def handle_summarize(self, data: dict):
        text = data.get("text", "")
        model_type = data.get("model", "bartpho")
        max_len = int(data.get("max_length", 150))
        min_len = int(data.get("min_length", 30))

        is_valid, msg, meta = VietnamesePreprocessor.validate_edge_cases(text)
        if not is_valid:
            self.send_json({"status": "error", "message": msg}, 400)
            return

        with Timer() as timer:
            if model_type == "lead1":
                summary = LeadKSummarizer(k=1).summarize(text)
            elif model_type == "lead3":
                summary = lead3_model.summarize(text)
            elif model_type == "tfidf":
                summary = tfidf_model.summarize(text)
            elif model_type == "chunking":
                res = chunking_model.summarize_map_reduce(text)
                summary = res.get("summary", "")
            else: # bartpho
                res = bartpho_model.summarize(text, max_target_length=max_len, min_target_length=min_len)
                summary = res.get("summary", "")

        fact_res = FactualityChecker.check_factuality(summary, text)
        input_words = len(VietnamesePreprocessor.tokenize_syllables(text))
        summary_words = len(VietnamesePreprocessor.tokenize_syllables(summary))

        self.send_json({
            "status": "success",
            "model_used": model_type,
            "summary": summary,
            "latency_ms": round(timer.elapsed_ms, 2),
            "input_words": input_words,
            "summary_words": summary_words,
            "compression_ratio": round((summary_words / input_words) * 100, 2) if input_words > 0 else 0,
            "factuality": fact_res
        })

    def handle_compare_all(self, data: dict):
        text = data.get("text", "")
        gold_summary = data.get("gold_summary", "")

        is_valid, msg, meta = VietnamesePreprocessor.validate_edge_cases(text)
        if not is_valid:
            self.send_json({"status": "error", "message": msg}, 400)
            return

        results = {}
        models = [
            ("Lead-1", LeadKSummarizer(k=1)),
            ("Lead-3", lead3_model),
            ("TF-IDF", tfidf_model),
            ("BARTpho", bartpho_model),
            ("Chunking", chunking_model)
        ]

        for name, m in models:
            with Timer() as timer:
                if name == "BARTpho":
                    s = m.summarize(text).get("summary", "")
                elif name == "Chunking":
                    s = m.summarize_map_reduce(text).get("summary", "")
                else:
                    s = m.summarize(text)

            fact = FactualityChecker.check_factuality(s, text)
            rouge_info = evaluator.evaluate_single(s, gold_summary) if gold_summary else None

            results[name] = {
                "summary": s,
                "latency_ms": round(timer.elapsed_ms, 2),
                "words": len(VietnamesePreprocessor.tokenize_syllables(s)),
                "factuality": fact,
                "rouge": rouge_info
            }

        self.send_json({
            "status": "success",
            "document_words": len(VietnamesePreprocessor.tokenize_syllables(text)),
            "results": results
        })

def run_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, SummarizationRequestHandler)
    print("=" * 70)
    print(f"🚀 [NLP-20 DEMO] Web Server đang chạy tại: http://127.0.0.1:{port}")
    print("   Nhấn Ctrl+C để dừng server.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng Web Server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
