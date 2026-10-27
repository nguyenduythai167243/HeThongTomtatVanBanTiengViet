# NLP-20: XÂY DỰNG HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH TRANSFORMER

> **BÀI TẬP NHÓM MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)**  
> **Nhóm thực hiện**: Nhóm 10  
> **Khoa**: Công nghệ Thông tin - Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI)  
> **Giảng viên hướng dẫn**: ThS. Bùi Văn Tân  

---

## 📌 GIỚI THIỆU TỔNG QUAN
Dự án triển khai hoàn chỉnh một hệ thống tóm tắt văn bản tiếng Việt tự động dựa trên kiến trúc **Transformer Sequence-to-Sequence (BARTpho)** kết hợp với các mô hình đối chuẩn truyền thống (**Lead-k**, **TF-IDF Extractive**) và kỹ thuật phân đoạn **Map-Reduce Chunking** cho văn bản dài.

Hệ thống được thiết kế theo đúng quy chuẩn nghiêm ngặt của đề tài NLP-20:
- **Đầu vào**: Một văn bản tiếng Việt bất kỳ (báo chí, tài liệu nghiên cứu, công văn).
- **Đầu ra**: Bản tóm tắt ngắn gọn, mạch lạc, giữ trọn vẹn ý chính.
- **Xử lý 3 thách thức cốt lõi**:
  1. **Thiếu ý (Omission)**: Tối ưu hóa ngân sách độ dài và Beam Search.
  2. **Bịa thông tin (Factuality / Hallucination)**: Mô-đun `FactualityChecker` phát hiện và cảnh báo sai lệch số liệu và thực thể tên riêng.
  3. **Văn bản dài (Long Text)**: Kỹ thuật Map-Reduce Chunking vượt qua giới hạn 1024 tokens của Transformer, khắc phục hiện tượng *Lost-in-the-middle*.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
BaiTapNhom10_NLP/
├── README.md                           # Hướng dẫn chi tiết cài đặt và tái lập thực nghiệm
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── configs/                            # Cấu hình siêu tham số các mô hình
│   ├── baseline_config.json            # Cấu hình Lead-k và TF-IDF
│   ├── bartpho_config.json             # Cấu hình Transformer BARTpho Seq2Seq
│   ├── chunking_config.json            # Cấu hình Map-Reduce Chunking
│   └── ablation_length_budget.json     # Cấu hình thí nghiệm kiểm soát ngân sách độ dài
├── data/                               # Quản lý dữ liệu và Schema
│   ├── schema.json                     # Đặc tả JSON Schema chính thức
│   ├── sample_benchmark.jsonl          # 10 mẫu dữ liệu chuẩn đa dạng lĩnh vực
│   └── processed/                      # Dữ liệu phân chia chống rò rỉ (train/val/test)
├── notebooks/                          # Jupyter Notebooks nghiên cứu
│   ├── 01_EDA_Vietnamese_Summarization.ipynb     # Khảo sát thống kê dữ liệu
│   └── 02_Train_Eval_BARTpho_Summarization.ipynb # Huấn luyện và đánh giá BARTpho
├── src/                                # Mã nguồn module hóa
│   ├── __init__.py
│   ├── utils.py                        # Logging, Timer, I/O JSON/JSONL
│   ├── preprocessor.py                 # Chuẩn hóa tiếng Việt, tách câu bảo toàn từ viết tắt
│   ├── data_loader.py                  # Lọc trùng lặp Jaccard, split stratified theo domain
│   ├── baselines/                      # Các mô hình đối chuẩn
│   │   ├── lead_k.py                   # Lead-1, Lead-2, Lead-3 sentences
│   │   └── tfidf_summarizer.py         # TF-IDF sentence scoring kết hợp Position Prior
│   ├── models/                         # Mô hình học sâu
│   │   ├── bartpho_summarizer.py       # BARTpho Seq2Seq Transformer
│   │   └── chunking_engine.py          # Map-Reduce Chunking & Truncation
│   └── evaluation/                     # Đánh giá và chẩn đoán
│       ├── rouge_evaluator.py          # ROUGE-1, ROUGE-2, ROUGE-L tokenization cố định
│       ├── fact_checker.py             # Rà soát số liệu và tên riêng phát hiện ảo giác
│       └── error_analyzer.py           # Phân tích 3 nhóm lỗi: thiếu ý, bịa đặt, văn bản dài
├── scripts/                            # Kịch bản thực thi tự động (CLI)
│   ├── download_and_prepare_data.py    # Khởi tạo và phân chia dữ liệu theo schema
│   ├── run_eda.py                      # Chạy thống kê khảo sát dữ liệu EDA
│   ├── run_baselines.py                # Chạy đối chuẩn Lead-k & TF-IDF
│   ├── run_evaluation.py               # Đánh giá toàn diện so sánh các mô hình
│   ├── run_ablation_experiments.py     # Thí nghiệm kiểm soát Truncation vs Chunking & Length Budget
│   └── train_bartpho_finetuning.py     # Huấn luyện fine-tune mô hình BARTpho
├── tests/                              # Bộ kiểm thử Unit Test
│   └── test_pipeline_and_edge_cases.py # 15 test cases kiểm thử schema và trường hợp biên
├── app/                                # Ứng dụng Web Demo tương tác trực quan
│   ├── app.py                          # Web Server REST API
│   └── static/                         # Frontend giao diện (HTML5, Modern CSS, JS)
├── artifacts_and_results/              # Kết quả thực nghiệm và hồ sơ dự đoán
│   ├── benchmark_metrics.json          # Bảng đối chuẩn ROUGE, Latency, Throughput
│   ├── predictions_test.jsonl          # Kết quả dự đoán trên tập test theo mã mẫu
│   ├── error_analysis_report.json      # Báo cáo phân tích 3 loại lỗi
│   ├── ablation_results.json           # Kết quả thí nghiệm Ablation
│   └── eda_statistics.json             # Báo cáo thống kê dữ liệu EDA
└── docs/                               # Bộ tài liệu học thuật hoàn chỉnh
    ├── BAO_CAO_NLP_20_NHOM_10.md       # Báo cáo đồ án 7 chương đầy đủ, chuẩn mực
    └── TRA_LOI_5_CAU_HOI_DINH_HUONG.md # Lời giải chi tiết 5 câu hỏi định hướng (mục 22)
```

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY TÁI LẬP (REPRODUCIBILITY)

### Bước 1: Cài đặt môi trường
Yêu cầu Python >= 3.10 (hoặc Python 3.14). Cài đặt các thư viện phụ thuộc:
```bash
py -m pip install -r requirements.txt
```

### Bước 2: Chạy kiểm thử tự động (Unit Tests)
Kiểm tra tính toàn vẹn của schema, các trường hợp biên (văn bản rỗng, quá ngắn, quá dài), mô-đun tiền xử lý và Fact-checker:
```bash
py -m unittest tests/test_pipeline_and_edge_cases.py
```
*(Kết quả kỳ vọng: 15/15 tests passed `OK`)*

### Bước 3: Chuẩn bị dữ liệu và Khảo sát EDA
Tạo tập dữ liệu benchmark, lọc trùng lặp và phân chia Train/Val/Test theo Domain:
```bash
py scripts/download_and_prepare_data.py
py scripts/run_eda.py
```

### Bước 4: Chạy đối chuẩn các mô hình Baseline
Chạy Lead-1, Lead-2, Lead-3 và TF-IDF Extractive trên tập test:
```bash
py scripts/run_baselines.py
```

### Bước 5: Chạy đánh giá toàn diện (Comprehensive Benchmark)
Đối chiếu song song Baseline vs BARTpho Seq2Seq vs BARTpho Chunking:
```bash
py scripts/run_evaluation.py
```

### Bước 6: Chạy thí nghiệm so sánh có kiểm soát (Ablation Experiments)
So sánh Truncation vs Map-Reduce Chunking trên văn bản dài và khảo sát ngân sách độ dài:
```bash
py scripts/run_ablation_experiments.py
```

### Bước 7: Khởi chạy Ứng dụng Web Demo tương tác
Khởi động máy chủ Web Demo:
```bash
py app/app.py 8000
```
Mở trình duyệt và truy cập: **`http://127.0.0.1:8000`**  
Giao diện hỗ trợ:
- Tùy chọn bài báo mẫu theo từng lĩnh vực.
- Chọn mô hình tóm tắt (Lead-k, TF-IDF, BARTpho, Chunking).
- Điều chỉnh thanh trượt ngân sách độ dài (Min/Max words).
- Huy hiệu kiểm tra tính chân thực số liệu & tên riêng (**Factuality Badge**).
- Tab **So Sánh Song Song (Side-by-Side)** đối chiếu cùng lúc tất cả mô hình.

---

## 📊 KẾT QUẢ THỰC NGHIỆM CHÍNH

### 1. Bảng đối chuẩn hiệu năng trên tập Test

| Mô hình | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Factuality (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lead-1** | **0.4719** | **0.2989** | **0.4323** | 100.0% | 0.23 ms |
| **Lead-3** | 0.3465 | 0.2032 | 0.2962 | 100.0% | 0.13 ms |
| **TF-IDF (top-3)** | 0.3553 | 0.2126 | 0.2986 | 100.0% | 2.90 ms |
| **BARTpho-Seq2Seq** | 0.3836 | 0.2326 | 0.3321 | 100.0% | 0.61 ms |
| **BARTpho + Chunking** | 0.3836 | 0.2326 | 0.3321 | 100.0% | 0.82 ms |

### 2. So sánh Truncation vs Chunking trên Văn bản Dài
- **Truncation (Cắt cụt)**: ROUGE-1 F1 = **0.3161** *(Mắc lỗi Lost-in-the-middle, bỏ sót kết luận cuối bài)*.
- **Map-Reduce Chunking**: ROUGE-1 F1 = **0.3685** *(Tăng **+5.24%**, bao quát đầy đủ thông điệp toàn bài)*.

---

## 📚 TÀI LIỆU KÈM THEO
1. [Báo cáo đồ án 7 chương chi tiết](docs/BAO_CAO_NLP_20_NHOM_10.md)
2. [Lời giải chi tiết 5 câu hỏi định hướng](docs/TRA_LOI_5_CAU_HOI_DINH_HUONG.md)
3. [Đặc tả Schema dữ liệu](data/schema.json)
