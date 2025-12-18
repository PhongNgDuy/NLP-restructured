## Lab 6 – Giới thiệu Transformers: Masked LM, Text Generation, Sentence Embedding

### 1. Mục tiêu

- Ôn lại các kiến thức cơ bản về kiến trúc Transformer thông qua 3 tác vụ điển hình:
  - Masked Language Modeling (Encoder-only – BERT).
  - Text Generation (Decoder-only – GPT).
  - Biểu diễn câu (Sentence Representation) bằng BERT.
- Thực hành sử dụng thư viện `transformers` (Hugging Face) và các pipeline có sẵn.

### 2. Môi trường & Cài đặt

- Sử dụng Python 3, thư viện chính:
  - `transformers`
  - `torch`
- Một số model được dùng:
  - `bert-base-uncased` cho masked language modeling và sentence embedding.
  - `gpt2` cho text generation.

### 3. Task 1 – Masked Language Modeling với BERT

**Mục tiêu:**  
Hiểu cách mô hình Encoder-only như BERT dự đoán token bị che (masked token) dựa trên ngữ cảnh hai chiều.

**Các bước chính:**

- Dùng `pipeline("fill-mask", model="bert-base-uncased")`.
- Câu đầu vào:  
  `Hanoi is the [MASK] of Vietnam.`
- Yêu cầu mô hình trả về Top-5 dự đoán (`top_k=5`).

**Kết quả chính:**

- Mô hình dự đoán:
  - `capital` với xác suất ~0.999 là từ phù hợp nhất.
  - Các từ còn lại như `center`, `birthplace`, `headquarters`, `city` có xác suất rất nhỏ.
- BERT phù hợp cho masked LM vì:
  - Encoder sử dụng self-attention hai chiều (bidirectional).
  - Thấy được cả ngữ cảnh **trước** (`Hanoi is the`) và **sau** (`of Vietnam`) vị trí `[MASK]`.

### 4. Task 2 – Text Generation với GPT (Decoder-only)

**Mục tiêu:**  
Thực hành sinh văn bản từ mô hình Decoder-only (GPT-2), hiểu đặc trưng “dự đoán từ tiếp theo”.

**Các bước chính:**

- Dùng `pipeline("text-generation", model="gpt2")`.
- Prompt:  
  `The best thing about learning NLP is`
- Thiết lập:
  - `max_length=50`
  - `num_return_sequences=3`

**Kết quả chính:**

- Mô hình sinh ra 3 đoạn văn khác nhau, đều:
  - Ngữ pháp tiếng Anh tương đối tốt.
  - Nội dung xoay quanh chủ đề “học NLP”, học hỏi, lợi ích, cách học…
- Do cơ chế sampling (và/hoặc greedy/beam), các lần chạy có thể cho kết quả khác nhau.
- GPT (Decoder-only) phù hợp cho text generation vì:
  - Được huấn luyện theo mục tiêu **Next Token Prediction**: Dự đoán token $x_t$ dựa trên chuỗi $x_1, \dots, x_{t-1}$.
  - Self-attention một chiều (causal) đảm bảo không “nhìn trước” tương lai.

### 5. Task 3 – Sentence Representation với BERT

**Mục tiêu:**  
Sinh vector biểu diễn cho cả câu dựa trên BERT, thông qua kỹ thuật Mean Pooling trên last hidden states.

**Các bước chính:**

- Dùng `AutoTokenizer` và `AutoModel` với `bert-base-uncased`.
- Câu đầu vào:  
  `"This is a sample sentence."`
- Tokenize:
  - `padding=True`, `truncation=True`, `return_tensors="pt"`.
- Forward qua BERT để lấy `last_hidden_state` (shape: `[batch_size, seq_len, hidden_size]`).
- Dùng `attention_mask` để **loại bỏ padding**:
  - Mở rộng `attention_mask` thành `(batch_size, seq_len, hidden_size)`.
  - Tính tổng embedding trên các token thật.
  - Chia cho số token thật để lấy **mean pooling**.

**Kết quả chính:**

- Vector biểu diễn câu có shape `torch.Size([1, 768])`.
- 768 là `hidden_size` của `bert-base-uncased`.
- Việc dùng `attention_mask` là cần thiết:
  - Nếu cộng/trung bình cả token padding, vector câu bị nhiễu và không phản ánh đúng nội dung thực.

### 6. Nhận xét & Kết luận

- **Masked LM (BERT)**:
  - Cho thấy sức mạnh của ngữ cảnh hai chiều trong việc điền từ bị che.
- **Text Generation (GPT)**:
  - Minh họa rõ ràng năng lực sinh văn bản tự do, nhưng cũng cho thấy tính không ổn định (kết quả mỗi lần chạy khác nhau).
- **Sentence Embedding với BERT**:
  - Mean Pooling là cách đơn giản nhưng hiệu quả để lấy vector câu từ `last_hidden_state`.
  - Chú ý xử lý `attention_mask` đúng cách.


