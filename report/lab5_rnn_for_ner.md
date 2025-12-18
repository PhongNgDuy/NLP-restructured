## Lab 5 – RNN cho bài toán Nhận dạng Thực thể Tên (NER)

### 1. Mục tiêu

- Xây dựng một pipeline hoàn chỉnh cho bài toán **Named Entity Recognition (NER)** trên bộ dữ liệu **CoNLL 2003**:
  - Tải và tiền xử lý dữ liệu bằng thư viện `datasets` của Hugging Face.
  - Xây dựng từ điển từ (vocabulary) và từ điển nhãn NER.
  - Tạo `Dataset` và `DataLoader` trong PyTorch cho token classification với padding động.
  - Định nghĩa mô hình **BiLSTM** đơn giản cho NER.
  - Huấn luyện và đánh giá mô hình, sau đó thử dự đoán trên câu mới.

### 2. Dữ liệu và tiền xử lý

- **Bộ dữ liệu**: `conll2003` từ Hugging Face Datasets:
  - Các split: `train`, `validation`, `test`.
  - Mỗi phần tử chứa:
    - `tokens`: danh sách từ trong câu.
    - `ner_tags`: danh sách id nhãn NER tương ứng với từng token.
- **Tải dữ liệu**:
  - Dùng `load_dataset("conll2003", trust_remote_code=True)`.
  - Lấy ra:
    - `train_sentences = dataset["train"]["tokens"]`
    - `train_ner_ids = dataset["train"]["ner_tags"]`
    - `val_sentences`, `val_ner_ids` tương tự cho `validation`.
- **Chuyển id → nhãn string**:
  - Dùng `dataset["train"].features["ner_tags"].feature.names` để lấy `id2tag`.
  - Chuyển toàn bộ `ner_tags` từ id sang string (`B-PER`, `I-PER`, `O`, …) để dễ debug và xây vocab nhãn.

### 3. Xây dựng từ điển (Vocabulary)

- Hàm `build_vocabs(sentences, tags)`:
  - **word_to_ix**:
    - Khởi tạo với:
      - `"<pad>"` → 0
      - `"<unk>"` → 1
    - Duyệt qua toàn bộ token trong tập train, gán index tăng dần cho mỗi từ mới.
  - **tag_to_ix**:
    - Duyệt qua toàn bộ nhãn NER dạng string trong tập train, gán index tăng dần cho mỗi nhãn mới.
- Kết quả ví dụ:
  - `Vocab size: 23625`
  - `Number of NER tags (without pad): 9`
- Sau đó thêm nhãn `<pad>` vào `tag_to_ix` để dùng cho padding nhãn trong batch:
  - Nếu chưa có `<pad>` thì gán index mới cho `<pad>`.

### 4. Dataset, DataLoader và Padding

#### 4.1 Lớp `NERDataset`

- Kế thừa `torch.utils.data.Dataset`, nhận vào:
  - `sentences: List[List[str]]`
  - `tags: List[List[str]]`
  - `word_to_ix`, `tag_to_ix`
- `__getitem__(idx)`:
  - Map từng từ trong câu sang index, dùng `"<unk>"` nếu từ ngoài vocab.
  - Map từng nhãn NER sang index.
  - Trả về:
    - `sentence_indices: LongTensor (T,)`
    - `tag_indices: LongTensor (T,)`

#### 4.2 Hàm `ner_collate_fn`

- Mục tiêu: pad các câu trong batch về cùng độ dài, đồng thời pad cả nhãn:
  - Tính `lengths` của từng câu và `max_len` trong batch.
  - Với mỗi cặp `(s, t)`:
    - Nếu ngắn hơn `max_len`:
      - Pad `s` bằng `pad_word_idx` (`<pad>`).
      - Pad `t` bằng `pad_tag_idx` (`<pad>`).
  - Trả về:
    - `batch_sentences: (B, T_max)`
    - `batch_tags: (B, T_max)`
    - `lengths_tensor: (B,)` – độ dài thực của từng câu.
- `DataLoader`:
  - `train_loader`: `batch_size=32`, `shuffle=True`, `collate_fn=collate_wrapper`.
  - `val_loader`: `batch_size=64`, `shuffle=False`.

### 5. Mô hình BiLSTM cho NER

- Lớp `SimpleBiLSTMForNER`:
  - **Embedding**:
    - `nn.Embedding(vocab_size, embedding_dim=128, padding_idx=pad_idx)`.
  - **LSTM**:
    - `nn.LSTM(embedding_dim, hidden_dim=256, num_layers=1, batch_first=True, bidirectional=True)`.
  - **Linear**:
    - `nn.Linear(hidden_dim * 2, tagset_size)` để map output mỗi token sang không gian nhãn.
- Forward:
  - Input: `x` (B, T), `lengths` (B,).
  - Dùng `pack_padded_sequence` với `enforce_sorted=False` để LSTM không tính trên padding.
  - Sau LSTM, dùng `pad_packed_sequence` để quay về dạng `(B, T, 2H)`.
  - Áp linear để thu được `logits` (B, T, C).

### 6. Huấn luyện và đánh giá

- Thiết lập:
  - Thiết bị: Colab GPU T4.
  - `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`.
  - `criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)`:
    - Bỏ qua các vị trí padding khi tính loss.
- Hàm `train_one_epoch`:
  - Lặp qua `train_loader`:
    - Forward → tính loss → backward → update.
    - Tính tổng loss chuẩn hóa theo số token **không phải padding**.
- Hàm `evaluate`:
  - Đặt `model.eval()`, `torch.no_grad()`.
  - Tính:
    - `avg_loss` trên token thật.
    - **Độ chính xác (accuracy) token-level**, bỏ qua padding:
      - Lấy `argmax` trên `logits_flat`.
      - So sánh với `tags_flat` tại các vị trí `mask = tags_flat != ignore_index`.

#### 6.1 Kết quả huấn luyện (validation)

- Sau 3 epoch, log trong notebook:

```text
Epoch 1/3 - train_loss: 0.5180 - val_loss: 0.3128 - val_acc: 0.9078
Epoch 2/3 - train_loss: 0.2127 - val_loss: 0.2092 - val_acc: 0.9395
Epoch 3/3 - train_loss: 0.1052 - val_loss: 0.1823 - val_acc: 0.9472
Best validation accuracy: 0.9472
```

- Nhận xét:
  - Loss train giảm đều qua các epoch.
  - Validation accuracy ~ **94.7%**, khá tốt cho một mô hình BiLSTM đơn giản, chưa dùng CRF hay pre-trained embeddings.

### 7. Dự đoán câu mới

- Hàm `predict_sentence(model, sentence, word_to_ix, ix_to_tag, device)`:
  - Tách câu theo khoảng trắng → danh sách token.
  - Map từng token sang index, dùng `<unk>` cho từ ngoài vocab.
  - Chạy forward (với độ dài đúng của câu).
  - Lấy `argmax` trên trục nhãn, sau đó map index nhãn → string bằng `ix_to_tag`.
  - Trả về danh sách `(token, nhãn_dự_đoán)`.

**Ví dụ trong notebook:**

- Câu:  
  `"VNU University is located in Hanoi"`
- Kết quả:

```text
VNU         B-ORG
University  I-ORG
is          O
located     O
in          O
Hanoi       B-LOC
```

- Nhận xét:
  - `"VNU University"` được gán nhãn tổ chức (ORG) đúng cấu trúc B/I.
  - `"Hanoi"` được nhận diện là địa danh (LOC).

### 8. Tổng kết

- Lab đã hiện thực đủ pipeline NER với:
  - Tiền xử lý dữ liệu CoNLL 2003.
  - Dataset/DataLoader có padding động.
  - Mô hình BiLSTM hai chiều cho token classification.
  - Huấn luyện, đánh giá và suy diễn trên câu mới.
- Mô hình đạt ~95% accuracy token-level trên tập validation, là baseline tốt cho:
  - So sánh với các mô hình nâng cao hơn (BiLSTM-CRF, BERT fine-tuning).
  - Thêm các cải tiến như dropout, pre-trained embeddings, hoặc tăng số layer.


