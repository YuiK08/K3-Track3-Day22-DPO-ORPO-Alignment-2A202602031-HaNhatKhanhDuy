# Báo Cáo Thực Hành: Preference Alignment (DPO/ORPO)
Họ tên: Hà Nhật Khánh Duy MSV: 2A202602031

*Báo cáo kết quả quá trình xử lý dữ liệu, cài đặt thuật toán và đánh giá mô hình.*

## 1. Phân tích & Làm sạch Dữ liệu (Dataset Analysis & Cleaning)

### Tổng quan quá trình load dữ liệu (Data Loading Summary)
- **Tổng số mẫu (examples) tải thành công**: `26` (Bao gồm 24 mẫu gốc + 2 mẫu dữ liệu tổng hợp thêm).
- **Các lỗi/vấn đề phát hiện (Validation issues)**: 
  - Dòng 1 trong file `sample_preferences.jsonl` bị lỗi cú pháp JSON do dấu ngoặc kép `""` quanh chữ "self-attention" không được escape đúng chuẩn.
  - Rủi ro tồn tại các prompt trùng lặp hoặc các câu trả lời Chosen/Rejected gần giống hệt nhau (chỉ khác khoảng trắng hoặc viết hoa/thường).
- **Các bước làm sạch (Cleaning steps taken)**: 
  - Sửa lỗi cú pháp thủ công dòng 1.
  - Bổ sung khối `try-except` vào `data.py` để tự động bỏ qua và cảnh báo log các dòng JSON hỏng.
  - Viết bộ lọc (Duplicate check) dựa trên tập `seen_prompts`.
  - Cập nhật Pydantic validator trong `schemas.py` dùng `.strip().lower()` để loại trừ chuẩn xác các cặp Chosen/Rejected trùng lặp về nội dung cốt lõi.

### Chiến lược phân chia dữ liệu (Split Strategy)
- **Tỉ lệ Train/Val**: `80/20`
- **Ngăn chặn rò rỉ dữ liệu (Leakage Prevention)**: Thay vì chia tuần tự từng hàng (row), thuật toán thực hiện **nhóm (grouping) theo prompt**. Sau đó shuffle dựa trên `seed=42` cố định (deterministic) để đảm bảo toàn bộ các cặp trả lời của cùng 1 prompt sẽ chỉ nằm ở tập Train hoặc tập Val, không bị rò rỉ chéo.

## 2. Cài đặt thuật toán: DPO & ORPO

### Lựa chọn mục tiêu (Objective Selection)
- **Thuật toán đã triển khai**: Cả **DPO** và **ORPO**.
  - *Lý do*: DPO chuẩn hóa tốt sự ưa thích tương đối (log-ratio), trong khi ORPO tận dụng việc tính Odds-ratio gộp chung trực tiếp vào quá trình SFT (Supervised Fine-Tuning) mà không cần mô hình tham chiếu (reference model).
- **Siêu tham số chính (Key Hyperparameters)**:
    - `beta` (DPO): `0.1`
    - `lambda_orpo` (ORPO): `0.1`

### Tính ổn định số học (Numerical Stability)
- **Thách thức (Challenges)**: Tính toán trực tiếp `log(sigmoid(x))` dễ dẫn đến hiện tượng underflow/overflow hoặc lỗi log(0) đối với các giá trị log-probability âm sâu. Với ORPO, việc tính `log(1 - exp(logp))` dễ gặp lỗi nếu p tiệm cận 1.
- **Giải pháp (Solutions)**: 
  - Áp dụng `np.logaddexp(0, -beta * logits)` cho DPO để tính toán an toàn thay cho hàm log-sigmoid gốc.
  - Sử dụng `np.log1p(-np.exp(np.clip(logps, -np.inf, -1e-7)))` cho ORPO để ghim cận chặn an toàn trước khi biến đổi log.

## 3. Kết quả đánh giá (Evaluation Results)

### Số liệu (Metrics)
| Metric | Value |
|---|---|
| Pairwise Accuracy | `84.62%` (Kết quả từ evaluate CLI với bộ số mô phỏng) |
| Final Loss (Mock/Train) | `DPO: ~0.693` / `ORPO: ~0.345` (Giá trị giả lập kỳ vọng đầu dải) |

### Đánh giá định tính (Qualitative Review)
- **Prompt**: `What is backpropagation in simple terms?`
- **Chosen Response**: `It is a way for neural networks to learn from mistakes by adjusting their internal settings backward from the output to the input.`
- **Rejected Response**: `It is an algorithm used for clustering data into groups.`
- **Model Preference**: Đang mô phỏng giả lập ngẫu nhiên (Mock), hệ thống tính điểm tie (hòa) bằng `0.5` hoặc ngẫu nhiên ưu tiên theo phân phối uniform. 

## 4. Bàn luận & Phân tích lỗi (Discussion & Failure Modes)

- **Điểm thành công (What went well?)**: 
  - Hoàn thành toàn bộ `TODO(student)` blocks. 
  - Pipeline chạy trơn tru từ quá trình chuẩn bị dữ liệu (Data Loading), chia tập (Split), tính toán hàm Loss phức tạp đến hàm Đánh giá (Evaluate).
  - Test framework (`pytest`) pass 100%.
- **Quan sát thiên kiến (Observed Bias)**: Do đang sử dụng bộ số Mock Trainer ngẫu nhiên, bias thực tế từ LLM chưa thể đánh giá sâu sắc. Tuy nhiên hệ thống Evaluation đã chứng minh cơ chế xử lý bằng điểm (Tie-breaking) tính là `0.5` giúp đánh giá khách quan hơn so với việc coi hòa là sai.
- **An toàn (Safety)**: Hệ thống Validation schema hoạt động cứng rắn, Dataset mẫu (Synthetic Preferences) sạch và đã vượt qua các bộ kiểm thử tự động, không chứa dữ liệu nhạy cảm (PII). Hỗ trợ sẵn cấu trúc để chạy regression prompts.
