# GIẢI ĐÁP CHI TIẾT 5 CÂU HỎI ĐỊNH HƯỚNG (MỤC 22)
**ĐỀ TÀI NLP-20: HỆ THỐNG TÓM TẮT VĂN BẢN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH TRANSFORMER**  
*Nhóm 10 - Khoa Công nghệ Thông tin - UNETI*

---

### Câu hỏi 1: Một tóm tắt có ROUGE cao vẫn sai dữ kiện ra sao?
**Trả lời:**
- **Bản chất của chỉ số ROUGE**: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) chỉ là độ đo dựa trên sự trùng khớp n-gram từ vựng (surface-level lexical overlap) và độ dài dãy con chung giữa bản tóm tắt sinh ra (hypothesis) và bản tóm tắt chuẩn (reference). ROUGE hoàn toàn **không có khả năng hiểu ngữ nghĩa, tính logic hay kiểm tra tính chân thực (factuality)** của dữ kiện.
- **Minh chứng sai lệch dữ kiện cụ thể**:
  - *Ví dụ 1 (Đảo ngược ngữ nghĩa / Phủ định)*:
    - **Bài gốc**: *"Bộ Y tế khẳng định hiện tại **chưa phát hiện** ca nhiễm biến chủng mới trong cộng đồng."*
    - **Tóm tắt chuẩn**: *"Hiện tại chưa phát hiện ca nhiễm biến chủng mới trong cộng đồng."*
    - **Tóm tắt mô hình sinh ra**: *"Bộ Y tế khẳng định hiện tại **đã phát hiện** ca nhiễm biến chủng mới trong cộng đồng."*
    - **Đánh giá**: Tóm tắt của mô hình chỉ khác bản chuẩn đúng 1 từ ("đã" thay vì "chưa"). ROUGE-1 F1 đạt tới **92.3%**, ROUGE-L F1 đạt **90%**, nhưng thông tin thực tế bị đảo ngược 180 độ, tạo nên tin giả cực kỳ nguy hiểm.
  - *Ví dụ 2 (Sai lệch số liệu / Thực thể tên riêng)*:
    - Thay thế *"tại Hà Nội"* thành *"tại TP. HCM"* hoặc *"chi phí 10 tỷ"* thành *"chi phí 100 tỷ"*. Các từ xung quanh đều trùng khớp nên điểm ROUGE vẫn rất cao, nhưng dữ kiện đã bị bịa đặt hoàn toàn (Factuality Hallucination).
- **Kết luận**: ROUGE cao là điều kiện cần về độ bao phủ từ vựng, nhưng không phải điều kiện đủ cho tính đúng đắn. Do đó, hệ thống NLP-20 bắt buộc phải tích hợp mô-đun `FactualityChecker` để kiểm soát số liệu và tên riêng.

---

### Câu hỏi 2: Tại sao baseline Lead-k hoặc tóm tắt trích xuất TF–IDF là một đối chứng cần thiết cho đề tài này?
**Trả lời:**
1. **Đặc trưng cấu trúc hình tháp ngược (Inverted Pyramid) trong báo chí**:
   - Trong phong cách báo chí truyền thông tiếng Việt, các phóng viên và biên tập viên luôn đặt thông tin quan trọng nhất (Who, What, Where, When, Why) ngay trong 1 đến 3 câu đầu tiên (Lead sentences). Thân bài và phần cuối chỉ cung cấp thêm các chi tiết phụ trợ hoặc giải thích bối cảnh.
   - Do đó, Lead-k (đặc biệt Lead-1 hoặc Lead-3) đạt điểm ROUGE rất cao một cách tự nhiên mà không cần bất kỳ tính toán học sâu phức tạp nào (trong thực nghiệm của nhóm, Lead-1 đạt ROUGE-1 F1 = 0.4719).
2. **Thiết lập ngưỡng sàn (Lower Bound) và kiểm tra tính thiết thực**:
   - Một mô hình Transformer Seq2Seq nặng hàng trăm triệu tham số, tiêu tốn nhiều tài nguyên GPU chỉ được coi là có ý nghĩa thực tế khi nó mang lại giá trị vượt trội (về độ cô đọng, khả năng tổng hợp liên câu, tính mạch lạc) so với việc chỉ cắt lấy câu đầu tiên (Lead-1) với chi phí 0.2 ms.
3. **Phát hiện lỗi suy giảm (Model Degeneration)**:
   - Nếu mô hình học sâu sinh ra câu lủng củng, lặp từ hoặc bịa đặt số liệu thì chất lượng sử dụng thực tế thậm chí còn kém hơn Lead-k hoặc TF-IDF. Baseline trích xuất (Extractive) luôn đảm bảo 100% ngữ pháp tiếng Việt chuẩn xác và không bao giờ bịa thông tin.

---

### Câu hỏi 3: Cách chia ở mục 14 ngăn loại rò rỉ nào, và còn giới hạn nào?
**Trả lời:**
1. **Loại rò rỉ dữ liệu được ngăn chặn**:
   - **Rò rỉ bài viết gần trùng (Near-Duplicate Leakage)**: Nhờ bước lọc trùng (Deduplication) bằng độ tương đồng Jaccard trước khi chia, các bài báo xào lại từ cùng một thông cáo báo chí không thể xuất hiện đồng thời ở cả tập Train và tập Test.
   - **Rò rỉ miền/chủ đề (Domain/Source Leakage)**: Việc phân chia theo phân tầng Domain (Stratified Domain Splitting) đảm bảo mô hình không bị "học vẹt" phong cách viết riêng của một tòa soạn hay sự kiện cụ thể ở tập Train rồi làm bài kiểm tra trên chính sự kiện đó ở tập Test.
   - **Rò rỉ tối ưu hóa (Hyperparameter Leakage)**: Toàn bộ việc tinh chỉnh siêu tham số (Beam size, Length budget, Learning rate) được quyết định trên tập Validation; tập Test hoàn toàn được giữ kín cho đến lượt đánh giá cuối cùng.
2. **Giới hạn còn tồn tại**:
   - **Trôi dạt phân phối theo thời gian (Temporal Concept Drift)**: Nếu các sự kiện mới phát sinh ngoài các chủ đề đã thu thập (ví dụ công nghệ mới, chính sách mới), mô hình vẫn có thể gặp khó khăn về từ vựng mới (Out-of-Vocabulary) và cách diễn đạt mới.
   - **Quy mô tập Test hẹp**: Khi số lượng mẫu kiểm thử chưa đủ lớn, một vài mẫu ngoại lai (outliers) có thể gây biến động điểm số thống kê (variance).

---

### Câu hỏi 4: Một ví dụ liên quan đến thiếu ý sẽ làm hệ thống thất bại ra sao?
**Trả lời:**
- **Tình huống thực tế**: Một tài liệu y tế hoặc quyết định pháp luật có cấu trúc điều kiện - ngoại lệ:
  - **Văn bản gốc**: *"UBND Thành phố quyết định cho phép toàn bộ các cơ sở kinh doanh dịch vụ ăn uống mở cửa hoạt động trở lại từ ngày 1/10, **nhưng nghiêm cấm hoạt động sau 21h và bắt buộc khách hàng phải quét mã QR khai báo y tế**."*
  - **Tóm tắt bị lỗi thiếu ý (Omission)**: *"UBND Thành phố cho phép các cơ sở ăn uống mở cửa hoạt động trở lại từ ngày 1/10."*
- **Hệ quả thất bại**:
  - Người đọc dựa vào bản tóm tắt sẽ hiểu rằng dịch vụ được mở cửa tự do không điều kiện, dẫn đến vi phạm quy định giờ đóng cửa (sau 21h) và không khai báo y tế.
  - Về mặt NLP, lỗi thiếu ý xảy ra khi mô hình dừng sinh câu quá sớm (do length penalty quá cao) hoặc chỉ bắt chước câu đầu mà bỏ qua mệnh đề đối lập (*"tuy nhiên", "nhưng", "ngoại trừ"*) ở vế sau. Đây là một thất bại nghiêm trọng trong các ứng dụng tóm tắt công văn, y tế và tài chính.

---

### Câu hỏi 5: Thực nghiệm so truncation/chunking hoặc ngân sách độ dài đầu ra có thể giúp kiểm tra giả thuyết nào?
**Trả lời:**
1. **Thực nghiệm Truncation vs Map-Reduce Chunking**:
   - **Giả thuyết 1 (Hiện tượng Lost-in-the-Middle & Truncation Bias)**: *"Khi tóm tắt văn bản dài vượt quá cửa sổ ngữ cảnh (1024 tokens), phương pháp cắt cụt (Truncation) sẽ làm mất hoàn toàn thông tin ở nửa sau bài viết (kết quả, giải pháp, kết luận), dẫn đến ROUGE Recall suy giảm rõ rệt."*
   - **Kết quả kiểm chứng**: Thực nghiệm của nhóm cho thấy Chunking Map-Reduce đạt **ROUGE-1 F1 = 0.3685**, cao hơn vượt bậc so với Truncation (**0.3161**, tăng +5.24%), chứng minh giả thuyết là hoàn toàn chính xác.
2. **Thực nghiệm Ngân sách Độ dài Đầu ra (Output Length Budget)**:
   - **Giả thuyết 2 (Trade-off giữa Tính Cô đọng và Độ Bao phủ)**: *"Ngân sách độ dài ngắn (Budget-Short: 20-60 từ) ép mô hình chỉ giữ lại nòng cốt câu, giúp tăng mạnh ROUGE Precision và ROUGE F1 khi nhãn tham chiếu là tóm tắt ngắn. Ngược lại, ngân sách dài (Budget-Long: >150 từ) làm tăng Recall nhưng khiến văn bản rườm rà, giảm Precision và dễ phát sinh nguy cơ bịa thông tin."*
   - **Kết quả kiểm chứng**: Bảng thực nghiệm khẳng định cấu hình Budget-Short đạt ROUGE-1 F1 cao nhất (0.4421 so với 0.3836 của Budget-Medium), xác thực mối tương quan chặt chẽ giữa ngân sách độ dài và hiệu năng tóm tắt.
