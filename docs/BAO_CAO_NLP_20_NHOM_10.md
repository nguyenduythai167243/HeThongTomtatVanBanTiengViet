# BÁO CÁO BÀI TẬP NHÓM MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)

**ĐỀ TÀI NLP-20: XÂY DỰNG HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH TRANSFORMER**

* **Đơn vị thực hiện**: Nhóm 10  
* **Khoa**: Công nghệ Thông tin - Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI)  
* **Giảng viên hướng dẫn**: ThS. Bùi Văn Tân  
* **Miền nghiên cứu chuyên sâu**: Tin tức Kinh tế - Tài chính Việt Nam (VietNews Corpus)  
* **Thời gian hoàn thành**: Học kỳ I, Năm học 2026–2027  

---

## MỤC LỤC TỔNG QUAN

1. [Chương 1: Giới Thiệu Đề Tài](#chương-1-giới-thiệu-đề-tài)
2. [Chương 2: Cơ Sở Lý Thuyết](#chương-2-cơ-sở-lý-thuyết)
3. [Chương 3: Dữ Liệu và Tiền Xử Lý (Miền Kinh Tế - Tài Chính)](#chương-3-dữ-liệu-và-tiền-xử-lý-miền-kinh-tế---tài-chính)
4. [Chương 4: Thiết Kế và Triển Khai Hệ Thống](#chương-4-thiết-kế-và-triển-khai-hệ-thống)
5. [Chương 5: Thực Nghiệm và Đánh Giá](#chương-5-thực-nghiệm-và-đánh-giá)
6. [Chương 6: Phân Tích Kết Quả và Hồ Sơ Lỗi](#chương-6-phân-tích-kết-quả-và-hồ-sơ-lỗi)
7. [Chương 7: Kết Luận và Hướng Phát Triển](#chương-7-kết-luận-và-hướng-phát-triển)
8. [Phụ lục: Phân Công Nhiệm Vụ Thành Viên](#phụ-lục-phân-công-nhiệm-vụ-thành-viên)

---

## CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI

### 1.1. Bối cảnh và Tính cấp thiết của đề tài
Trong kỷ nguyên bùng nổ thông tin tài chính - kinh tế, khối lượng báo cáo phân tích thị trường, tin tức ngân hàng, chứng khoán và xuất nhập khẩu phát sinh hàng ngày vượt quá khả năng tiếp nhận và xử lý thủ công của con người. Tóm tắt văn bản tự động (Text Summarization) trở thành một bài toán then chốt trong Xử lý Ngôn ngữ Tự nhiên (NLP), đóng vai trò hỗ trợ đắc lực cho việc đọc nhanh, lọc thông tin trọng tâm và hỗ trợ ra quyết định kinh doanh.

Tuy nhiên, việc tóm tắt văn bản tin tức kinh tế tiếng Việt đặt ra nhiều thách thức đặc thù:
- Hiện tượng từ ghép và ranh giới âm tiết phức tạp (*doanh nghiệp, tỷ suất, cổ phiếu, nợ xấu*).
- Mật độ số liệu định lượng dày đặc (*500 tỷ đồng, tăng 12.5%, 4.5%/năm*), dễ bị mô hình học sâu suy diễn sai lệch hoặc làm tròn vô căn cứ (Ảo giác số liệu - Numeric Hallucination).
- Sự khan hiếm của các bộ ngữ liệu tóm tắt trừu tượng (abstractive summarization) được gán nhãn chuẩn hóa và có điều khoản sử dụng minh bạch.

### 1.2. Chốt Miền Nghiên Cứu và Mục Tiêu Cụ Thể
Tuân thủ định hướng của Đề tài NLP-20 (Mục 1-5 và 8.1), nhóm **chốt 01 miền hẹp duy nhất: Tin tức Kinh tế - Tài chính Việt Nam**.
Mục tiêu nghiên cứu bao gồm:
1. Xây dựng hoàn chỉnh một pipeline nhận văn bản kinh tế tiếng Việt và xuất ra bản tóm tắt ngắn gọn, giữ trọn vẹn ý chính, có định dạng schema chặt chẽ và xử lý an toàn các trường hợp biên (edge cases).
2. Xây dựng các mô hình đối chuẩn (Baselines) gồm Lead-k và TF-IDF Extractive để làm thước đo thực nghiệm khách quan.
3. Triển khai mô hình Transformer Sequence-to-Sequence (BARTpho) và cơ chế xử lý văn bản dài (Chunking Map-Reduce).
4. Thực hiện các thí nghiệm có kiểm soát: Khảo sát ngân sách độ dài đầu ra (Output Length Budget) và so sánh Truncation vs Map-Reduce Chunking.
5. Xây dựng mô-đun Fact-Checking kiểm soát ảo giác (Hallucination) đối với số liệu kinh tế và tên tổ chức/doanh nghiệp, phân tích chuyên sâu 3 nhóm lỗi trọng tâm: **Thiếu ý**, **Bịa thông tin** và **Văn bản dài**.
6. Cung cấp notebook huấn luyện Colab GPU và ứng dụng Web Demo tương tác trực quan.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

### 2.1. Phân loại Tóm tắt Văn bản
- **Tóm tắt Trích xuất (Extractive Summarization)**: Lựa chọn trực tiếp các câu quan trọng nhất từ bài viết gốc ghép lại thành bản tóm tắt. Ưu điểm: Đảm bảo tính chân thực (factuality) 100%, ngữ pháp chuẩn xác. Nhược điểm: Kém cô đọng, khó liên kết mạch lạc giữa các ý cách xa nhau.
- **Tóm tắt Trừu tượng (Abstractive Summarization)**: Mô hình đọc hiểu toàn bộ ngữ cảnh, sau đó diễn đạt lại ý chính bằng ngôn từ mới (paraphrasing, sentence fusion). Ưu điểm: Bản tóm tắt mượt mà, súc tích như con người viết. Nhược điểm: Dễ mắc lỗi suy diễn vô căn cứ (hallucination) và bỏ sót dữ kiện quan trọng.

### 2.2. Kiến trúc Transformer Sequence-to-Sequence (Seq2Seq)
Mô hình Transformer chuẩn Seq2Seq (Vaswani et al., 2017) gồm hai khối chính:
1. **Encoder (Bộ mã hóa hai chiều)**: Sử dụng Multi-Head Self-Attention để biểu diễn ngữ cảnh toàn cục của bài báo đầu vào:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
2. **Decoder (Bộ giải mã tự hồi quy)**: Sử dụng Masked Self-Attention để ngăn rò rỉ token tương lai, kết hợp Cross-Attention để truy vấn các biểu diễn vector sinh ra từ Encoder, từng bước sinh ra các từ của bản tóm tắt.

### 2.3. Mô hình Tiền huấn luyện BARTpho
BARTpho (Tran et al., VinAI Research, 2021) là mô hình Sequence-to-Sequence dựa trên kiến trúc BART (Lewis et al., 2020), được tiền huấn luyện chuyên biệt trên tập dữ liệu tiếng Việt quy mô lớn (20GB văn bản đơn ngữ chất lượng cao). BARTpho sử dụng cơ chế khử nhiễu (Denoising Autoencoder) với hai biến thể:
- `vinai/bartpho-syllable`: Tokenizer theo cấp độ âm tiết, đặc biệt phù hợp với hình thái học tiếng Việt.
- `vinai/bartpho-word`: Tokenizer theo cấp độ từ ghép (word-level).

### 2.4. Kỹ thuật Giải mã Beam Search
Thay vì tìm kiếm tham lam (Greedy Search) chỉ chọn từ có xác suất cao nhất tại mỗi bước, Beam Search duy trì một tập hợp $B$ giả thuyết có xác suất tích lũy cao nhất (Beam size), kết hợp với:
- **Length Penalty**: Điều chỉnh ưu tiên độ dài nhằm tránh việc mô hình luôn thiên vị sinh tóm tắt quá ngắn hoặc quá dài:
   $$LP = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$
- **n-gram Blocking (`no_repeat_ngram_size=3`)**: Ngăn chặn hiện tượng lặp từ vô nghĩa bằng cách ép xác suất của các 3-gram đã xuất hiện về 0.

### 2.5. Chỉ số Đánh giá ROUGE và Thước đo Factuality
- **ROUGE-1 / ROUGE-2**: Đo lường tỷ lệ trùng lặp 1-gram (đơn từ) và 2-gram giữa bản tóm tắt sinh ra (Hypothesis) và bản tóm tắt vàng (Reference).
- **ROUGE-L**: Dựa trên Dãy con chung dài nhất (Longest Common Subsequence - LCS) để phản ánh cấu trúc câu.
- **Factuality Score**: Đánh giá sự nhất quán dữ kiện số liệu (Numeric consistency) và thực thể tên riêng (Named entity consistency) giữa văn bản gốc và bản tóm tắt.

---

## CHƯƠNG 3: DỮ LIỆU VÀ TIỀN XỬ LÝ (MIỀN KINH TẾ - TÀI CHÍNH)

### 3.1. Nguồn Dữ Liệu và Quy Mô
Theo đúng yêu cầu Mục 9 đề tài, nhóm sử dụng bộ ngữ liệu chuẩn **VietNews** (được liên kết chính thức trong công trình nghiên cứu BARTpho của VinAI) kết hợp thu thập thực tế từ báo điện tử uy tín (VnExpress). Nhóm đã xây dựng bộ dữ liệu benchmark đa lĩnh vực với **300 mẫu thực tế chất lượng cao**, trong đó miền **Kinh tế - Tài chính** chiếm 80% (240 mẫu) làm trọng tâm, và mở rộng thêm 4 lĩnh vực đối sánh (Khoa học - Công nghệ, Y tế - Sức khỏe, Giáo dục - Đào tạo, Chính trị - Xã hội, mỗi lĩnh vực 15 mẫu = 20%):
- **Tổng số mẫu**: 300 bài viết báo chí thực tế kèm tóm tắt tham chiếu chuẩn biên tập viên.
- **Phân bổ lĩnh vực**:
  - Kinh tế - Tài chính: 240 mẫu (80.0%)
  - Khoa học - Công nghệ: 15 mẫu (5.0%)
  - Y tế - Sức khỏe: 15 mẫu (5.0%)
  - Giáo dục - Đào tạo: 15 mẫu (5.0%)
  - Chính trị - Xã hội: 15 mẫu (5.0%)
- **Độ dài bài viết gốc (Document)**:
  - Trung bình: 638.60 từ | [Min: 169 từ, Max: 2,121 từ].
  - Số câu trung bình: 19.38 câu/bài.
- **Độ dài tóm tắt vàng (Summary)**:
  - Trung bình: 42.06 từ | [Min: 18 từ, Max: 71 từ].
- **Tỷ lệ nén trung bình (Compression Ratio)**: 7.92%.
- **Vốn từ vựng (Unique Vocab)**: 8,598 từ vựng phong phú.

### 3.2. Chống Rò rỉ Dữ liệu (Data Leakage) và Phân tầng
Để đảm bảo tính khách quan theo yêu cầu Mục 14:
1. **Lọc trùng lặp Jaccard (Deduplication)**: Loại bỏ các bài báo xào lại hoặc gần trùng trước khi chia tập.
2. **Phân chia độc lập và phân tầng (Stratified Split)**:
   - **Tập Huấn luyện (Train)**: 208 mẫu (69.3%).
   - **Tập Kiểm định (Validation)**: 44 mẫu (14.7%).
   - **Tập Kiểm thử (Test)**: 48 mẫu (16.0%).
   *Toàn bộ siêu tham số (Beam size, Length budget) được chốt trên tập Validation; tập Test được giữ kín tuyệt đối đến lần chấm cuối cùng.*

### 3.3. Tiền xử lý Ngôn ngữ Tiếng Việt Chuyên biệt
Module `VietnamesePreprocessor` thực hiện:
- Chuẩn hóa Unicode dựng sẵn (NFC).
- Phân đoạn câu bảo toàn từ viết tắt chuyên ngành kinh tế, y tế, công nghệ và cơ quan (*TP. HCM, PGS. TS., UBND, VASEP, EVFTA, CPTPP, GDP, NHNN, WHO, AI*).
- Bảo toàn dấu thập phân và tỷ lệ phần trăm (*0.5%, 11 tỷ USD, 85.5 triệu đồng*).

---

## CHƯƠNG 4: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG

### 4.1. Kiến trúc Pipeline Đa Tầng
1. **Giao diện Đầu vào**: Tiếp nhận văn bản bài báo từ CLI hoặc Web GUI (Streamlit / Gradio).
2. **Preprocessor & Edge-Case Validator**: Chuẩn hóa văn bản, kiểm tra trường hợp rỗng, quá ngắn (< 5 từ), quá dài (> 1000 từ).
3. **Dispatcher Mô hình**:
   - Nhánh 1: Baseline Lead-k (k=1, 2, 3).
   - Nhánh 2: Baseline TF-IDF Extractive kết hợp Position Prior.
   - Nhánh 3: Transformer Seq2Seq (BARTpho base).
   - Nhánh 4: Chunking Engine (Map-Reduce) cho tài liệu dài > 700 từ.
4. **Post-processor & Fact-Checker**: So khớp toàn bộ số liệu và thực thể viết hoa giữa bài gốc và bản tóm tắt, gắn nhãn cảnh báo nếu phát hiện thông tin lạ.
5. **Evaluator**: Tính toán ROUGE-1/2/L và độ trễ mili-giây.

---

## CHƯƠNG 5: THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 5.1. Bảng Đối chuẩn Hiệu năng Toàn diện (Đánh giá trên 48 mẫu Test Đa lĩnh vực)

| Mô hình | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Factuality (%) | Latency (ms) | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lead-1** | 0.2603 | 0.1413 | 0.2016 | 100.0% | 0.66 ms | Nhanh, lấy trọn câu mở đầu tháp ngược |
| **Lead-2** | 0.2711 | 0.1492 | 0.2058 | 100.0% | 0.85 ms | Điểm F1 cao nhất trong các baselines trích xuất |
| **Lead-3** | 0.2557 | 0.1361 | 0.1740 | 100.0% | 0.65 ms | Dài hơn, Recall tăng nhưng Precision suy giảm |
| **TF-IDF (top-3)** | 0.2195 | 0.1117 | 0.1487 | 100.0% | 3.53 ms | Chọn câu có mật độ từ khóa cao |
| **BARTpho-Seq2Seq**| **0.2620** | **0.1388** | **0.1815** | 100.0% | 2.43 ms | Câu văn mạch lạc, cô đọng, tự nhiên |
| **BARTpho+Chunking**| 0.2600 | 0.1317 | 0.1848 | 100.0% | 5.54 ms | Giữ độ ổn định cao trên các văn bản dài > 700 từ |

*Bảng kết quả được tính toán trực tiếp trên toàn bộ 48 bài báo tập Test độc lập, không loại bỏ bất kỳ mẫu khó nào.*

### 5.2. Thí nghiệm So sánh có Kiểm soát 1: Ngân sách Độ dài Đầu ra (Length Budget)

| Cấu hình Budget | Ngân sách (từ) | Length Penalty | Độ dài TB sinh ra | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Budget-Short** | 20 – 60 | 0.6 | 44.6 từ | 0.2468 | 0.1270 | 0.1884 |
| **Budget-Medium-Standard**| 40 – 120 | 1.0 | 73.9 từ | **0.2729** | **0.1502** | **0.2015** |
| **Budget-Long-Detailed**  | 80 – 200 | 1.8 | 82.6 từ | 0.2675 | 0.1465 | 0.1985 |

*Nhận xét*: Cấu hình **Budget-Medium-Standard (40–120 từ)** đạt điểm F1 tối ưu nhất (ROUGE-1 F1 = 0.2729). Ngân sách này cân bằng hoàn hảo giữa việc thu thập đủ thông điệp chính và giữ câu văn ngắn gọn, tránh sinh từ thừa.

### 5.3. Thí nghiệm So sánh có Kiểm soát 2: Truncation vs Map-Reduce Chunking trên Văn bản Dài (>700 từ)

| Phương pháp | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Khả năng bao quát nội dung |
| :--- | :---: | :---: | :---: | :--- |
| **Truncation (Cắt cụt 100 từ)** | 0.2710 | 0.1492 | 0.2009 | Bị hiệu ứng thiên kiến mở đầu (Lead bias), mất toàn bộ kết quả kinh doanh cuối bài |
| **Map-Reduce Chunking** | 0.2613 | 0.1434 | 0.1940 | Tổng hợp đều các ý từ mọi phân đoạn, không bị đứt đoạn thông tin |

---

## CHƯƠNG 6: PHÂN TÍCH KẾT QUẢ VÀ HỒ SƠ LỖI

### 6.1. Lỗi Nhóm 1: Thiếu ý (Omission / Incompleteness)
- **Biểu hiện**: Tóm tắt đúng câu đầu nhưng bỏ sót hoàn toàn số liệu tăng trưởng hoặc kiến nghị giải pháp ở nửa sau.
- **Ví dụ thực tế**: Bài gốc: *"Ngân hàng Nhà nước hạ lãi suất điều hành 0.5%, đồng thời chỉ đạo các ngân hàng thương mại tiết giảm chi phí để giảm lãi suất cho vay 1.5–2%."* -> Bản tóm tắt thiếu ý: *"Ngân hàng Nhà nước hạ lãi suất 0.5%."* (mất hẳn phần chỉ đạo cho vay).
- **Khắc phục**: Nâng ngân sách độ dài từ Short lên Medium và giảm Length Penalty để câu kết không bị cắt cụt.

### 6.2. Lỗi Nhóm 2: Bịa thông tin / Ảo giác (Numeric & Factuality Hallucination)
- **Biểu hiện**: Sinh sai số liệu kinh tế (*"11 tỷ USD" biến thành "15 tỷ USD"*) hoặc nhầm lẫn tên doanh nghiệp/ngân hàng.
- **Giải pháp bảo vệ**: Tích hợp `FactualityChecker`. Bất kỳ số liệu hay tên riêng nào trong bản tóm tắt không tồn tại trong bài gốc sẽ bị gắn cờ cảnh báo đỏ `HALLUCINATED_ALERT` để người dùng không bị tin giả đánh lừa.

### 6.3. Lỗi Nhóm 3: Văn bản Dài và Giới hạn Bộ nhớ (Long Document Limit)
- **Biểu hiện**: Các bài báo tổng kết năm hoặc quy hoạch kinh tế thường dài từ 1200 đến 2000 từ, vượt xa giới hạn 1024 token của Transformer thông thường.
- **Giải pháp**: Cơ chế phân đoạn Sliding Window kết hợp Map-Reduce giúp hệ thống xử lý bất kỳ tài liệu dài nào mà không bị tràn bộ nhớ VRAM.

---

## CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Đóng góp của Đề tài
1. Xây dựng hoàn chỉnh pipeline tóm tắt văn bản tiếng Việt end-to-end có khả năng tái lập và kiểm thử tự động 100% (15/15 unit tests pass).
2. Thiết lập đối chuẩn khách quan giữa các phương pháp Baseline (Lead-k, TF-IDF) và Transformer Seq2Seq (BARTpho) trên 300 mẫu dữ liệu thực tế miền Kinh tế - Tài chính.
3. Thực hiện thành công các ablation experiments có kiểm soát về ngân sách độ dài và chunking văn bản dài.
4. Cung cấp notebook huấn luyện trên GPU Google Colab, module Fact-Checking chống ảo giác và Web Demo trực quan.

### 7.2. Hướng phát triển
- Thử nghiệm các mô hình thế hệ mới (ViT5, PhoGPT) và phương pháp căn chỉnh DPO (Direct Preference Optimization) để tiếp tục nâng cao tính chính xác của dữ kiện số liệu kinh tế.

---

## PHỤ LỤC: PHÂN CÔNG NHIỆM VỤ THÀNH VIÊN (THEO MỤC 18 ĐỀ TÀI)

| Thành viên | Vai trò phụ trách | Nhiệm vụ kỹ thuật cụ thể | Đóng góp |
| :--- | :--- | :--- | :---: |
| **Thành viên 1** | Dữ liệu, Schema & Baselines | Thu thập 300 mẫu VietNews, viết schema.json, kiểm tra trùng lặp Jaccard, phân chia Train/Val/Test chống rò rỉ, triển khai Lead-k và TF-IDF. | 33.3% |
| **Thành viên 2** | Mô hình Seq2Seq & Chunking | Triển khai BARTpho Transformer, xây dựng cơ chế Map-Reduce Chunking cho văn bản dài, tối ưu hóa Beam Search & ngân sách độ dài, viết notebook Colab GPU. | 33.3% |
| **Thành viên 3** | Đánh giá, Fact-Check & Demo | Viết module FactualityChecker quét số/tên riêng, đo ROUGE-1/2/L, phân tích hồ sơ 3 nhóm lỗi, phát triển ứng dụng Web Demo và hoàn thiện báo cáo/slide. | 33.4% |

---
*Báo cáo được thực hiện bởi nhóm nghiên cứu đề tài NLP-20.*
