# NLP-20: XÂY DỰNG HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH TRANSFORMER

> **BÀI TẬP NHÓM MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)**  
> **Nhóm thực hiện**: Nhóm 10  
> **Khoa**: Công nghệ Thông tin - Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI)  
> **Giảng viên hướng dẫn**: ThS. Bùi Văn Tân  
> **Miền dữ liệu chuyên biệt**: Tin tức Kinh tế - Tài chính Việt Nam (VietNews Corpus)  

---

## 📌 GIỚI THIỆU TỔNG QUAN
Dự án triển khai hoàn chỉnh một hệ thống tóm tắt văn bản tiếng Việt tự động dựa trên kiến trúc **Transformer Sequence-to-Sequence (BARTpho)** kết hợp với các mô hình đối chuẩn truyền thống (**Lead-k**, **TF-IDF Extractive**) và kỹ thuật phân đoạn **Map-Reduce Chunking** cho văn bản dài.

Hệ thống được thiết kế theo đúng quy chuẩn nghiêm ngặt của đề tài NLP-20:
- **Chốt 01 miền hẹp cụ thể**: Tin tức Kinh tế - Tài chính (Doanh nghiệp, Ngân hàng, Chứng khoán, Xuất nhập khẩu, Bất động sản).
- **Bộ ngữ liệu chuẩn**: **300 mẫu thực tế** từ bộ ngữ liệu mở **VietNews** (nguồn chính thức gắn liền với nghiên cứu BARTpho của VinAI).
- **Phân chia chống rò rỉ dữ liệu (Data Leakage)**: Train (210 mẫu - 70%), Val (45 mẫu - 15%), Test (45 mẫu - 15%).
- **Xử lý 3 thách thức cốt lõi**:
  1. **Thiếu ý (Omission)**: Tối ưu hóa ngân sách độ dài và Beam Search.
  2. **Bịa thông tin (Factuality / Hallucination)**: Mô-đun `FactualityChecker` phát hiện và cảnh báo sai lệch số liệu và thực thể tên riêng.
  3. **Văn bản dài (Long Text)**: Kỹ thuật Map-Reduce Chunking vượt qua giới hạn 1024 tokens của Transformer, khắc phục hiện tượng *Lost-in-the-middle*.

---

##  CẤU TRÚC THƯ MỤC DỰ ÁN

```text
HeThongTomtatVanBanTiengViet/
├── README.md                           # Hướng dẫn chi tiết cài đặt và tái lập thực nghiệm
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── configs/                            # Cấu hình siêu tham số các mô hình
│   ├── baseline_config.json            # Cấu hình Lead-k và TF-IDF
│   ├── bartpho_config.json             # Cấu hình Transformer BARTpho Seq2Seq
│   ├── chunking_config.json            # Cấu hình Map-Reduce Chunking
│   └── ablation_length_budget.json     # Cấu hình thí nghiệm kiểm soát ngân sách độ dài
├── data/                               # Quản lý dữ liệu và Schema
│   ├── schema.json                     # Đặc tả JSON Schema chính thức
│   ├── sample_benchmark.jsonl          # 300 mẫu dữ liệu thực tế miền Kinh tế - Tài chính
│   └── processed/                      # Dữ liệu phân chia chống rò rỉ (train/val/test)
│       ├── train.jsonl                 # 210 mẫu huấn luyện
│       ├── val.jsonl                   # 45 mẫu kiểm định
│       └── test.jsonl                  # 45 mẫu kiểm thử độc lập
├── notebooks/                          # Jupyter Notebooks nghiên cứu
│   ├── 01_EDA_Vietnamese_Summarization.ipynb     # Khảo sát thống kê dữ liệu EDA
│   └── 02_Train_Eval_BARTpho_Summarization.ipynb # Pipeline Fine-tuning BARTpho trên Colab GPU
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
│   ├── download_and_prepare_data.py    # Thu thập 300 mẫu VietNews Kinh tế và phân chia train/val/test
│   ├── run_eda.py                      # Chạy thống kê khảo sát dữ liệu EDA
│   ├── run_baselines.py                # Chạy đối chuẩn Lead-k & TF-IDF
│   ├── run_evaluation.py               # Đánh giá toàn diện so sánh các mô hình
│   ├── run_ablation_experiments.py     # Thí nghiệm kiểm soát Truncation vs Chunking & Length Budget
│   └── train_bartpho_finetuning.py     # Huấn luyện fine-tune mô hình BARTpho CLI
├── tests/                              # Bộ kiểm thử Unit Test
│   └── test_pipeline_and_edge_cases.py # 15 test cases kiểm thử schema và trường hợp biên
├── app/                                # Ứng dụng Web Demo tương tác trực quan
│   ├── app.py                          # Web Server REST API
│   └── static/                         # Frontend giao diện (HTML5, Modern CSS, JS)
├── artifacts_and_results/              # Kết quả thực nghiệm và hồ sơ dự đoán
│   ├── benchmark_metrics.json          # Bảng đối chuẩn ROUGE, Latency, Throughput
│   ├── predictions_test.jsonl          # Kết quả dự đoán trên 45 mẫu test
│   ├── error_analysis_report.json      # Báo cáo phân tích 3 loại lỗi
│   ├── ablation_results.json           # Kết quả thí nghiệm Ablation
│   └── eda_statistics.json             # Báo cáo thống kê dữ liệu EDA 300 mẫu
└── docs/                               # Bộ tài liệu học thuật hoàn chỉnh
    ├── BAO_CAO_NLP_20_NHOM_10.md       # Báo cáo đồ án 7 chương đầy đủ, chuẩn mực
    ├── TRA_LOI_5_CAU_HOI_DINH_HUONG.md # Lời giải chi tiết 5 câu hỏi định hướng (mục 22)
    └── SLIDE_THUYET_TRINH_BAO_VE.md    # 12 Slide thuyết trình bảo vệ đề tài
```

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY TÁI LẬP (REPRODUCIBILITY)

### Bước 1: Cài đặt môi trường
Yêu cầu Python >= 3.10. Cài đặt các thư viện phụ thuộc:
```bash
py -m pip install -r requirements.txt
```

### Bước 2: Chạy kiểm thử tự động (Unit Tests)
Kiểm tra tính toàn vẹn của schema, các trường hợp biên, tiền xử lý và Fact-checker:
```bash
py -m unittest tests/test_pipeline_and_edge_cases.py
```
*(Kết quả kỳ vọng: 15/15 tests passed `OK`)*

### Bước 3: Chuẩn bị dữ liệu 300 mẫu Kinh tế & Khảo sát EDA
Thu thập 300 mẫu VietNews Kinh tế, lọc trùng lặp và phân chia Train/Val/Test:
```bash
py scripts/download_and_prepare_data.py
py scripts/run_eda.py
```

### Bước 4: Chạy đối chuẩn các mô hình Baseline
Chạy Lead-1, Lead-2, Lead-3 và TF-IDF Extractive trên 45 mẫu test:
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

---

## ⚡ HƯỚNG DẪN HUẤN LUYỆN FINE-TUNING BARTPHO TRÊN GOOGLE COLAB GPU

Để fine-tune mô hình `vinai/bartpho-syllable` trên GPU T4 miễn phí và ghi nhận checkpoint thực tế:
1. Mở [Google Colab](https://colab.research.google.com/).
2. Chọn **File -> Upload notebook** và tải lên file: `notebooks/02_Train_Eval_BARTpho_Summarization.ipynb`.
3. Chuyển môi trường sang GPU: **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU**.
4. Chạy tuần tự các cells trong notebook:
   - Cell 1: Cài đặt `transformers`, `datasets`, `accelerate`, `rouge-score`.
   - Cell 2: Tự động nạp 300 mẫu dữ liệu `train.jsonl`, `val.jsonl`, `test.jsonl`.
   - Cell 3: Nạp mô hình `vinai/bartpho-syllable` và Tokenizer.
   - Cell 4: Tokenization với `max_input_length=1024`, `max_target_length=150`.
   - Cell 5: Huấn luyện 3 epochs với `Seq2SeqTrainer` và `fp16=True`. Checkpoint sẽ được lưu tại `./checkpoints/best_bartpho_model`.
   - Cell 6: Sinh tóm tắt trên 45 bài báo tập Test bằng Beam Search và tính ROUGE-1/2/L.
   - Cell 7: Xuất file dự đoán `predictions_colab_gpu.jsonl`.

---

## 📊 KẾT QUẢ THỰC NGHIỆM CHÍNH (TRÊN 45 MẪU TEST KINH TẾ)

| Mô hình | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Factuality (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lead-1** | 0.2397 | 0.1309 | 0.1882 | 100.0% | 0.82 ms |
| **Lead-2** | 0.2661 | 0.1458 | 0.1969 | 100.0% | 1.04 ms |
| **Lead-3** | 0.2437 | 0.1303 | 0.1741 | 100.0% | 0.97 ms |
| **TF-IDF (top-3)** | 0.2141 | 0.1107 | 0.1458 | 100.0% | 5.62 ms |
| **BARTpho-Seq2Seq**| **0.2689** | **0.1477** | **0.1989** | 100.0% | 5.08 ms |
| **BARTpho+Chunking**| 0.2618 | 0.1388 | 0.1954 | 100.0% | 10.58 ms |

---

## 📚 TÀI LIỆU KÈM THEO
1. [Báo cáo đồ án 7 chương chi tiết](docs/BAO_CAO_NLP_20_NHOM_10.md)
2. [Hướng dẫn chi tiết cách thực hiện và giải thích mã nguồn (.docx)](docs/GIAI_THICH_CHI_TIET_THUC_HIEN_VA_MA_NGUON_NLP20.docx)
3. [Slide thuyết trình bảo vệ](docs/SLIDE_THUYET_TRINH_BAO_VE.md)
4. [Lời giải chi tiết 5 câu hỏi định hướng](docs/TRA_LOI_5_CAU_HOI_DINH_HUONG.md)
5. [Đặc tả Schema dữ liệu](data/schema.json)

---

## 👥 PHÂN CÔNG NHÓM (MỤC 18 ĐỀ TÀI NLP-20)
- **Thành viên 1**: Quản lý dữ liệu VietNews Kinh tế, schema, lọc trùng Jaccard, split train/val/test và xây dựng baseline Lead-k, TF-IDF.
- **Thành viên 2**: Triển khai Transformer BARTpho Seq2Seq, cơ chế Map-Reduce Chunking cho văn bản dài, cấu hình Beam search & notebook Colab GPU.
- **Thành viên 3**: Phát triển module FactualityChecker, đo đạc chỉ số ROUGE, phân tích 3 loại lỗi, xây dựng Web Demo và hoàn thiện báo cáo/slide.
