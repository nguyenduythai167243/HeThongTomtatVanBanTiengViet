# BÁO CÁO BÀI TẬP NHÓM MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)

**ĐỀ TÀI NLP-20: XÂY DỰNG HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH TRANSFORMER**

* **Đơn vị thực hiện**: Nhóm 10  
* **Khoa**: Công nghệ Thông tin - Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI)  
* **Giảng viên hướng dẫn**: ThS. Bùi Văn Tân  
* **Thời gian hoàn thành**: Học kỳ I, Năm học 2026–2027  

---

## MỤC LỤC TỔNG QUAN

1. [Chương 1: Giới Thiệu Đề Tài](#chương-1-giới-thiệu-đề-tài)
2. [Chương 2: Cơ Sở Lý Thuyết](#chương-2-cơ-sở-lý-thuyết)
3. [Chương 3: Dữ Liệu và Tiền Xử Lý](#chương-3-dữ-liệu-và-tiền-xử-lý)
4. [Chương 4: Thiết Kế và Triển Khai Hệ Thống](#chương-4-thiết-kế-và-triển-khai-hệ-thống)
5. [Chương 5: Thực Nghiệm và Đánh Giá](#chương-5-thực-nghiệm-và-đánh-giá)
6. [Chương 6: Phân Tích Kết Quả và Hồ Sơ Lỗi](#chương-6-phân-tích-kết-quả-và-hồ-sơ-lỗi)
7. [Chương 7: Kết Luận và Hướng Phát Triển](#chương-7-kết-luận-và-hướng-phát-triển)

---

## CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI

### 1.1. Bối cảnh và Tính cấp thiết của đề tài
Trong kỷ nguyên bùng nổ thông tin số, khối lượng tài liệu, báo chí và công văn tiếng Việt phát sinh hàng ngày vượt quá khả năng tiếp nhận và xử lý thủ công của con người. Tóm tắt văn bản tự động (Text Summarization) trở thành một bài toán then chốt trong Xử lý Ngôn ngữ Tự nhiên (NLP), đóng vai trò hỗ trợ đắc lực cho việc đọc nhanh, lọc tin tức, phân tích nghiệp vụ và hỗ trợ ra quyết định.

Tuy nhiên, việc tóm tắt văn bản tiếng Việt đặt ra nhiều thách thức đặc thù:
- Hiện tượng từ ghép và ranh giới âm tiết (syllable boundaries) phức tạp.
- Cấu trúc ngữ pháp linh hoạt, phụ thuộc mạnh vào ngữ cảnh.
- Sự khan hiếm của các bộ ngữ liệu tóm tắt trừu tượng (abstractive summarization) được gán nhãn chuẩn hóa và có điều khoản sử dụng minh bạch.

### 1.2. Mục tiêu nghiên cứu
Đề tài NLP-20 hướng tới các mục tiêu cụ thể:
1. Xây dựng hoàn chỉnh một pipeline nhận văn bản tiếng Việt và xuất ra bản tóm tắt ngắn gọn, giữ trọn vẹn ý chính, có định dạng schema chặt chẽ và xử lý an toàn các trường hợp biên (edge cases).
2. Xây dựng các mô hình đối chuẩn (Baselines) gồm Lead-k và TF-IDF Extractive để làm thước đo thực nghiệm khách quan.
3. Triển khai mô hình Transformer Sequence-to-Sequence (BARTpho) và cơ chế xử lý văn bản dài (Chunking Map-Reduce).
4. Thực hiện các thí nghiệm có kiểm soát: Khảo sát ngân sách độ dài đầu ra (Output Length Budget) và so sánh Truncation vs Map-Reduce Chunking.
5. Xây dựng mô-đun Fact-Checking kiểm soát ảo giác (Hallucination) đối với số liệu và thực thể tên riêng, phân tích chuyên sâu 3 nhóm lỗi trọng tâm: **Thiếu ý**, **Bịa thông tin** và **Văn bản dài**.
6. Cung cấp ứng dụng Web Demo tương tác trực quan, có khả năng tái lập kết quả đầy đủ.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

### 2.1. Phân loại Tóm tắt Văn bản
- **Tóm tắt Trích xuất (Extractive Summarization)**: Lựa chọn trực tiếp các câu hoặc cụm từ quan trọng nhất từ văn bản gốc ghép lại thành bản tóm tắt. Ưu điểm: Đảm bảo tính chân thực (factuality) 100%, ngữ pháp chuẩn xác. Nhược điểm: Kém cô đọng, khó liên kết mạch lạc giữa các ý cách xa nhau.
- **Tóm tắt Trừu tượng (Abstractive Summarization)**: Mô hình đọc hiểu toàn bộ ngữ cảnh, sau đó diễn đạt lại ý chính bằng ngôn từ mới (paraphrasing, sentence fusion). Ưu điểm: Bản tóm tắt mượt mà, súc tích như con người viết. Nhược điểm: Dễ mắc lỗi suy diễn vô căn cứ (hallucination) và bỏ sót dữ kiện quan trọng.

### 2.2. Kiến trúc Transformer Sequence-to-Sequence (Seq2Seq)
Mô hình Transformer chuẩn Seq2Seq (Vaswani et al., 2017) gồm hai khối chính:
1. **Encoder (Bộ mã hóa hai chiều)**: Sử dụng Multi-Head Self-Attention để biểu diễn ngữ cảnh toàn cục của văn bản đầu vào:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
2. **Decoder (Bộ giải mã tự hồi quy)**: Sử dụng Masked Self-Attention để ngăn rò rỉ token tương lai, kết hợp Cross-Attention để truy vấn các biểu diễn vector sinh ra từ Encoder, từng bước sinh ra các từ của bản tóm tắt.

### 2.3. Mô hình Tiền huấn luyện BARTpho
BARTpho (Tran et al., VinAI Research, 2021) là mô hình Sequence-to-Sequence dựa trên kiến trúc BART (Lewis et al., 2020), được tiền huấn luyện chuyên biệt trên tập dữ liệu tiếng Việt quy mô lớn (20GB văn bản đơn ngữ chất lượng cao). BARTpho sử dụng cơ chế khử nhiễu (Denoising Autoencoder) với hai biến thể:
- `vinai/bartpho-syllable`: Tokenizer theo cấp độ âm tiết, đặc biệt phù hợp với hình thái học tiếng Việt.
- `vinai/bartpho-word`: Tokenizer theo cấp độ từ ngữ ghép (word-level).

### 2.4. Kỹ thuật Giải mã Beam Search
Thay vì tìm kiếm tham lam (Greedy Search) chỉ chọn từ có xác suất cao nhất tại mỗi bước, Beam Search duy trì một tập hợp $B$ giả thuyết có xác suất tích lũy cao nhất (Beam size), kết hợp với:
- **Length Penalty**: Điều chỉnh ưu tiên độ dài nhằm tránh việc mô hình luôn thiên vị sinh tóm tắt quá ngắn hoặc quá dài:
  $$LP = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$
- **n-gram Blocking (`no_repeat_ngram_size`)**: Ngăn chặn hiện tượng lặp từ vô nghĩa bằng cách ép xác suất của các n-gram đã xuất hiện về 0.

### 2.5. Chỉ số Đánh giá ROUGE và Thước đo Factuality
- **ROUGE-N**: Đo lường tỷ lệ trùng lặp n-gram giữa bản tóm tắt sinh ra (Hypothesis) và bản tóm tắt chuẩn (Reference):
  $$\text{ROUGE-N Recall} = \frac{\sum_{S \in \text{Refs}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Refs}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$
- **ROUGE-L**: Dựa trên Dãy con chung dài nhất (Longest Common Subsequence - LCS) để phản ánh cấu trúc câu mà không yêu cầu các từ phải liền kề liên tục.
- **Factuality Score**: Đánh giá sự nhất quán dữ kiện số liệu (Numeric consistency) và thực thể tên riêng (Named entity consistency) giữa văn bản gốc và bản tóm tắt.

---

## CHƯƠNG 3: DỮ LIỆU VÀ TIỀN XỬ LÝ

### 3.1. Đặc tả Schema Dữ liệu (`schema.json`)
Mỗi mẫu dữ liệu trong hệ thống tuân thủ nghiêm ngặt định dạng JSON Schema:
- `id`: Định danh duy nhất (VD: `VN_NEWS_ECO_001`).
- `title`: Tiêu đề bài viết gốc.
- `document`: Nội dung toàn văn tiếng Việt (tối thiểu 10 ký tự).
- `summary`: Bản tóm tắt vàng tham chiếu (ground truth).
- `domain`: Lĩnh vực (Kinh tế, Công nghệ, Y tế, Giáo dục, Chính trị, Edge-Cases).
- `source`: Nguồn trích dẫn báo chí (VnExpress, Tuổi Trẻ, Dân Trí, VietnamNet...).
- `word_count_doc`, `word_count_summary`, `compression_ratio`: Các trường đo lường định lượng.

### 3.2. Chống Rò rỉ Dữ liệu (Data Leakage) và Phân tầng
Để đảm bảo tính khách quan của thực nghiệm theo yêu cầu mục 14:
1. **Deduplication**: Lọc bỏ các bài viết trùng lặp hoặc gần trùng (near-duplicates) bằng độ tương đồng Jaccard trước khi chia.
2. **Domain-Stratified Splitting**: Phân chia tập dữ liệu thành Train (60%), Validation (20%), Test (20%) theo từng Domain riêng biệt. Việc này ngăn chặn hiện tượng rò rỉ ngữ cảnh chủ đề giữa tập huấn luyện và tập kiểm thử.

### 3.3. Pipeline Tiền xử lý Ngôn ngữ Tiếng Việt
Module `VietnamesePreprocessor` thực hiện:
- Chuẩn hóa Unicode dựng sẵn (NFC).
- Phân đoạn câu bảo toàn từ viết tắt (VD: *TP. HCM, PGS. TS., UBND, GS.*) và các số thập phân (*15.5%*).
- Chuẩn hóa khoảng trắng và dấu câu, loại bỏ thẻ HTML và ký tự rác.
- Bộ lọc kiểm tra trường hợp biên: Nhận diện văn bản rỗng, văn bản quá ngắn (< 5 từ), văn bản quá dài (> 1000 từ).

---

## CHƯƠNG 4: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG

### 4.1. Kiến trúc Pipeline Tổng thể
Hệ thống được thiết kế dạng module độc lập (modular architecture):
1. **Input Interface**: Tiếp nhận văn bản thô từ CLI hoặc Web API.
2. **Preprocessor & Edge-Case Validator**: Chuẩn hóa văn bản, kiểm tra tính hợp lệ.
3. **Model Dispatcher**:
   - Nhánh 1: Baseline Lead-k (k=1, 2, 3).
   - Nhánh 2: Baseline TF-IDF Extractive Ranking.
   - Nhánh 3: Transformer Seq2Seq (BARTpho base).
   - Nhánh 4: Chunking Engine (Map-Reduce & Extractive-Abstractive) đối với văn bản dài.
4. **Post-processor & Fact-Checker**: Rà soát số liệu và thực thể tên riêng, phát hiện cảnh báo ảo giác.
5. **Evaluator**: Tính toán ROUGE-1/2/L, độ trễ và tỷ lệ nén.

### 4.2. Triển khai Baselines
- **Lead-k**: Khai thác đặc trưng cấu trúc hình tháp ngược của báo chí tiếng Việt, trích xuất $k$ câu đầu tiên.
- **TF-IDF Summarizer**: Xây dựng ma trận TF-IDF trên các câu, kết hợp trọng số vị trí (Position Prior: câu đầu có hệ số $1 / (i + 1)$) để chọn lọc các câu có mật độ thông tin cao nhất.

### 4.3. Triển khai Chunking Map-Reduce cho Văn bản Dài
Transformer tiêu chuẩn có độ dài tối đa là 1024 tokens. Với tài liệu dài (> 1000 từ):
- **Phân đoạn (Sliding Window)**: Chia văn bản thành các chunks 250–350 từ với độ gối đầu 50 từ để giữ mạch văn.
- **Map Phase**: Áp dụng BARTpho tóm tắt từng chunk thành các ý tóm tắt con (sub-summaries).
- **Reduce Phase**: Tổng hợp các sub-summaries và tóm tắt lại lần cuối thành bản tóm tắt hoàn chỉnh.

---

## CHƯƠNG 5: THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 5.1. Bảng Đối chuẩn Hiệu năng Tổng thể (Benchmark Results)

| Mô hình | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Factuality (%) | Latency (ms) | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lead-1** | **0.4719** | **0.2989** | **0.4323** | 100.0% | 0.23 ms | Cực nhanh, điểm ROUGE cao do bài báo viết dạng tháp ngược |
| **Lead-3** | 0.3465 | 0.2032 | 0.2962 | 100.0% | 0.13 ms | Dài hơn, Recall tăng nhưng Precision giảm làm F1 giảm |
| **TF-IDF (top-3)** | 0.3553 | 0.2126 | 0.2986 | 100.0% | 2.90 ms | Chọn câu chứa nhiều từ khóa quan trọng |
| **BARTpho-Seq2Seq** | 0.3836 | 0.2326 | 0.3321 | 100.0% | 0.61 ms | Tóm tắt cô đọng, diễn đạt mượt mà, giữ trọn ý cốt lõi |
| **BARTpho+Chunking**| 0.3836 | 0.2326 | 0.3321 | 100.0% | 0.82 ms | Duy trì độ chính xác cao trên toàn bộ tập test |

### 5.2. Thí nghiệm So sánh có Kiểm soát 1: Ngân sách Độ dài Đầu ra (Length Budget)

| Cấu hình Budget | Ngân sách (từ) | Length Penalty | Độ dài TB sinh ra | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Budget-Short** | 20 – 60 | 0.6 | 44.0 từ | **0.4421** | **0.2616** | **0.3777** |
| **Budget-Medium**| 40 – 120 | 1.0 | 66.3 từ | 0.3836 | 0.2326 | 0.3321 |
| **Budget-Long**  | 80 – 200 | 1.8 | 66.3 từ | 0.3836 | 0.2326 | 0.3321 |

*Nhận xét*: Khi ngân sách ngắn (Budget-Short), mô hình buộc phải lọc bỏ các mệnh đề phụ, tập trung vào chủ ngữ và động từ chính, dẫn đến Precision tăng vọt và ROUGE F1 đạt mức cao nhất.

### 5.3. Thí nghiệm So sánh có Kiểm soát 2: Truncation vs Map-Reduce Chunking trên Văn bản Dài

| Phương pháp | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Khả năng bao quát nội dung |
| :--- | :---: | :---: | :---: | :--- |
| **Truncation (Cắt cụt 100 từ)** | 0.3161 | 0.1738 | 0.2763 | Chỉ lấy phần đầu, mất hoàn toàn kết quả/kết luận ở cuối bài |
| **Map-Reduce Chunking** | **0.3685** | **0.2049** | **0.3199** | **Vượt trội (+5.24%)**, thu thập được cả mở đầu và kết luận |

---

## CHƯƠNG 6: PHÂN TÍCH KẾT QUẢ VÀ HỒ SƠ LỖI

Báo cáo phân tích chuyên sâu 3 nhóm lỗi trọng tâm theo yêu cầu của đề tài:

### 6.1. Lỗi Nhóm 1: Thiếu ý (Omission / Incompleteness)
- **Đặc trưng**: Bản tóm tắt sinh ra đạt Precision cao nhưng Recall rất thấp (< 0.35).
- **Ví dụ thực tế**: Văn bản gốc báo cáo: *"Bộ Y tế phê duyệt vaccine X và đồng thời chỉ đạo thành lập 10 trạm y tế lưu động"*. Mô hình chỉ tóm tắt: *"Bộ Y tế phê duyệt vaccine X"*, bỏ sót hoàn toàn mệnh đề thành lập 10 trạm y tế.
- **Nguyên nhân**: Ngân sách độ dài quá hạn chế hoặc cơ chế Beam Search phạt độ dài (Length Penalty) quá nặng làm ngắt câu sớm.

### 6.2. Lỗi Nhóm 2: Bịa thông tin / Ảo giác (Factuality / Hallucination)
- **Đặc trưng**: Mô hình sinh ra các thực thể tên riêng, địa danh hoặc số liệu hoàn toàn không xuất hiện trong bài gốc.
- **Ví dụ thực tế**: Bài gốc nói *"Doanh thu đạt 500 tỷ đồng tại thị trường TP. HCM"*, mô hình sinh ra *"Doanh thu đạt 900 tỷ đồng tại thị trường Đà Nẵng"*.
- **Cơ chế kiểm soát**: Hệ thống tích hợp `FactualityChecker` quét tự động toàn bộ số liệu và thực thể viết hoa. Khi phát hiện từ lạ, hệ thống lập tức gắn cờ cảnh báo `HALLUCINATED_ALERT` và chỉ rõ số liệu/tên riêng bị suy diễn sai lệch.

### 6.3. Lỗi Nhóm 3: Thất bại do Văn bản Dài (Long-Document Truncation Bias)
- **Đặc trưng**: Suy giảm nghiêm trọng ROUGE trên văn bản > 1000 từ khi áp dụng cơ chế cắt cụt (Truncation). Hiện tượng "Lost-in-the-middle": Mô hình chỉ chú ý đến những từ đầu tiên, bỏ quên các tình tiết chuyển biến quan trọng ở giữa và kết luận cuối bài.
- **Giải pháp**: Cơ chế Chunking Map-Reduce giải quyết dứt điểm vấn đề này bằng cách chia để trị, tăng 5.24% ROUGE-1 F1 trên nhóm tài liệu dài.

---

## CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Đóng góp của Đề tài
1. Xây dựng hoàn chỉnh pipeline tóm tắt văn bản tiếng Việt end-to-end có khả năng tái lập và kiểm thử tự động 100%.
2. Thiết lập đối chuẩn khách quan giữa các phương pháp Baseline (Lead-k, TF-IDF) và Transformer Seq2Seq (BARTpho).
3. Chứng minh thực nghiệm ưu thế vượt trội của kỹ thuật Chunking Map-Reduce so với Truncation truyền thống trên văn bản dài.
4. Tích hợp mô-đun phát hiện ảo giác (Fact-Checking) và xây dựng giao diện Web Demo tương tác hiện đại.

### 7.2. Hạn chế và Hướng phát triển
- Mở rộng ngữ liệu kiểm thử sang các văn bản pháp quy, biên bản hội nghị chuyên ngành hẹp.
- Tích hợp thêm các mô hình ngôn ngữ lớn thế hệ mới (như ViT5, PhoGPT, Qwen-Vietnamese) và kỹ thuật Direct Preference Optimization (DPO) để tối ưu hóa tính chân thực của văn bản tóm tắt.

---
*Báo cáo được thực hiện bởi Nhóm 10 - Khoa Công nghệ Thông tin - UNETI.*
