# SLIDE THUYẾT TRÌNH BẢO VỆ BÀI TẬP NHÓM NLP

**ĐỀ TÀI NLP-20: XÂY DỰNG HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH TRANSFORMER**  
* **Nhóm thực hiện**: Nhóm 10  
* **Khoa**: Công nghệ Thông tin - Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI)  
* **Giảng viên hướng dẫn**: ThS. Bùi Văn Tân  
* **Miền dữ liệu chuyên biệt**: Tin tức Kinh tế - Tài chính Việt Nam (VietNews Corpus)  

---

## SLIDE 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ
- **Bài toán**: Tự động sinh tóm tắt ngắn giữ trọn vẹn ý chính từ văn bản tiếng Việt thô.
- **Tính cấp thiết**: Bùng nổ tin tức tài chính, báo cáo thị trường, ngân hàng -> Hỗ trợ đọc nhanh, chắt lọc dữ kiện kinh doanh.
- **3 Thách thức cốt lõi (theo đề bài NLP-20)**:
  1. **Thiếu ý (Omission)**: Mất mệnh đề trọng tâm hoặc kiến nghị giải pháp.
  2. **Bịa thông tin (Factuality / Hallucination)**: Sai lệch số liệu kinh tế và tên riêng.
  3. **Văn bản dài (Long text)**: Vượt quá giới hạn 1024 token của Transformer.

---

## SLIDE 2: MỤC TIÊU & PHẠM VI NGHIÊN CỨU
- **Chốt 01 miền hẹp cụ thể**: Tin tức Kinh tế - Tài chính Việt Nam.
- **Dữ liệu chuẩn**: 300 mẫu thực tế từ bộ ngữ liệu **VietNews** (nguồn chính thức gắn với nghiên cứu BARTpho của VinAI).
- **Phân chia chống rò rỉ dữ liệu (Data Leakage)**:
  - Train: 210 mẫu (70%)
  - Validation: 45 mẫu (15%)
  - Test: 45 mẫu (15%)
  - Lọc trùng lặp Jaccard trước khi chia tập.

---

## SLIDE 3: CƠ SỞ LÝ THUYẾT & KIẾN TRÚC MÔ HÌNH
- **Kiến trúc Transformer Sequence-to-Sequence (Seq2Seq)**:
  - **Encoder hai chiều**: Tự chú ý đa đầu (Multi-Head Self-Attention).
  - **Decoder tự hồi quy**: Masked Self-Attention + Cross-Attention.
- **Mô hình tiền huấn luyện BARTpho**:
  - Checkpoint `vinai/bartpho-syllable` được pre-trained trên 20GB văn bản tiếng Việt chất lượng cao.
- **Cơ chế giải mã tối ưu**: Beam Search (Beam size = 4), Length Penalty, no_repeat_ngram_size = 3.

---

## SLIDE 4: THIẾT KẾ HỆ THỐNG VÀ ĐA TẦNG PIPELINE
1. **Input**: Nhận bài viết báo chí tiếng Việt thô.
2. **Preprocessor & Edge Cases Validator**: Chuẩn hóa Unicode NFC, giữ nguyên viết tắt (TP. HCM, GDP, VASEP, NHNN), phân đoạn câu.
3. **Model Dispatcher**:
   - Baseline 1: Lead-k (k=1, 2, 3)
   - Baseline 2: TF-IDF Extractive + Position Prior
   - Phương án chính: BARTpho Seq2Seq
   - Nâng cao: Map-Reduce Chunking cho tài liệu dài (>700 từ).
4. **FactualityChecker**: Rà soát đối chiếu số liệu và tên riêng phát hiện ảo giác.
5. **Output**: Bản tóm tắt chuẩn định dạng kèm chỉ số tin cậy.

---

## SLIDE 5: GIẢI PHÁP CHO VĂN BẢN DÀI (CHUNKING MAP-REDUCE)
- **Vấn đề**: Transformer bị giới hạn context window 1024 tokens; Truncation (cắt cụt) gây mất 100% kết quả và kiến nghị cuối bài.
- **Thuật toán Map-Reduce Chunking**:
  - **Sliding Window**: Chia văn bản thành các chunks 250–350 từ với độ gối đầu 50 từ.
  - **Map Phase**: BARTpho tóm tắt song song từng chunk thành các ý tóm tắt con.
  - **Reduce Phase**: Tổng hợp các ý con và tóm tắt lại lần cuối thành bản tóm tắt hoàn chỉnh.

---

## SLIDE 6: MÔ-ĐUN KIỂM SOÁT ẢO GIÁC (FACT-CHECKING)
- **Tại sao cần Fact-Checking?**: ROUGE chỉ đo trùng lặp từ vựng, không phát hiện được câu bịa đặt số liệu (ví dụ: *11 tỷ USD* thành *15 tỷ USD*).
- **Cơ chế hoạt động của `FactualityChecker`**:
  - Trích xuất toàn bộ số liệu định lượng và thực thể tên riêng trong Summary.
  - Đối chiếu ngược lại Document gốc.
  - Nếu xuất hiện dữ kiện lạ -> Lập tức gắn nhãn cảnh báo đỏ `HALLUCINATED_ALERT`.

---

## SLIDE 7: KẾT QUẢ ĐỐI CHUẨN HIỆU NĂNG (BENCHMARK)
*Đo đạc trên 48 bài báo tập Test độc lập đa lĩnh vực (Kinh tế, Công nghệ, Y tế, Giáo dục, Xã hội):*

| Mô hình | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Factuality | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lead-1** | 0.2603 | 0.1413 | 0.2016 | 100% | 0.66 ms |
| **Lead-2** | 0.2711 | 0.1492 | 0.2058 | 100% | 0.85 ms |
| **Lead-3** | 0.2557 | 0.1361 | 0.1740 | 100% | 0.65 ms |
| **TF-IDF (top-3)** | 0.2195 | 0.1117 | 0.1487 | 100% | 3.53 ms |
| **BARTpho-Seq2Seq** | **0.2620** | **0.1388** | **0.1815** | 100% | 2.43 ms |
| **BARTpho+Chunking**| 0.2600 | 0.1317 | 0.1848 | 100% | 5.54 ms |

*Nhận xét: Lead-2 có điểm từ vựng n-gram cao nhờ văn phong báo chí tháp ngược; BARTpho-Seq2Seq sinh câu tự nhiên, tóm tắt tổng hợp mạch lạc và bảo toàn thông tin ổn định qua Chunking trên văn bản dài.*

---

## SLIDE 8: THÍ NGHIỆM SO SÁNH CÓ KIỂM SOÁT (ABLATION)
1. **Thí nghiệm Ngân sách Độ dài Đầu ra**:
   - Budget-Short (20-60 từ): F1 = 0.2468 (quá ngắn, dễ thiếu ý)
   - **Budget-Medium (40-120 từ)**: **F1 = 0.2729** (tối ưu nhất)
   - Budget-Long (80-200 từ): F1 = 0.2675 (rườm rà, giảm Precision)
2. **Thí nghiệm Truncation vs Chunking trên Văn bản Dài**:
   - Truncation: ROUGE cao ở đầu nhưng mất hoàn toàn phần kết luận.
   - Map-Reduce Chunking: Đảm bảo bao phủ toàn diện 100% nội dung bài báo.

---

## SLIDE 9: HỒ SƠ LỖI VÀ BÀI HỌC KINH NGHIỆM (ERROR PROFILE)
- **Lỗi thiếu ý**: Xử lý bằng cách tối ưu hóa Beam Search và Length penalty vừa phải (1.0).
- **Lỗi ảo giác số liệu**: Xử lý bằng bộ lọc FactualityChecker trước khi trả về cho người dùng.
- **Lỗi thiên vị văn bản dài (Lead Bias)**: Cần kết hợp kỹ thuật Chunking phân tầng để giữ mạch logic.

---

## SLIDE 10: DEMO HỆ THỐNG WEB TƯƠNG TÁC
- Xây dựng Web GUI tương tác thời gian thực (Python REST API + HTML5/CSS3).
- **Tính năng nổi bật**:
  - Cho phép tùy chọn mô hình: Lead-1, Lead-3, TF-IDF, BARTpho Seq2Seq, Chunking Map-Reduce.
  - Tùy chỉnh ngân sách độ dài linh hoạt (Short, Medium, Long).
  - Trực quan hóa số từ, tỷ lệ nén, thời gian xử lý và **Hộp kiểm định Fact-Checking**.

---

## SLIDE 11: TRẢ LỜI CÁC CÂU HỎI ĐỊNH HƯỚNG CỐT LÕI
1. **ROUGE cao vẫn sai sự thật?**: Do ROUGE chỉ đo n-gram trùng khớp, thay đổi 1 từ phủ định (*chưa* -> *đã*) làm sai hoàn toàn dữ kiện nhưng ROUGE vẫn đạt >90%.
2. **Tại sao Lead-k là baseline bắt buộc?**: Báo chí tiếng Việt viết theo cấu trúc hình tháp ngược; Lead-k là ngưỡng sàn hiệu quả để đo lường giá trị thực tế của Deep Learning.
3. **Cách chia chống rò rỉ?**: Lọc trùng lặp Jaccard ngăn near-duplicate leakage; phân tầng domain ngăn style leakage.

---

## SLIDE 12: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- **Kết quả đạt được**:
  - Hoàn thành đầy đủ 23/23 mục theo yêu cầu đề tài NLP-20.
  - Bộ mã nguồn modular, có test cases tự động (15/15 tests pass), khả năng tái lập 100%.
  - Dữ liệu 300 mẫu chuẩn miền Kinh tế - Tài chính và notebook huấn luyện Colab GPU sẵn sàng.
- **Hướng phát triển**: Mở rộng với các mô hình ngôn ngữ lớn (LLM) và kỹ thuật DPO.

---
*Cảm ơn Thầy và các bạn đã chú ý lắng nghe!*
