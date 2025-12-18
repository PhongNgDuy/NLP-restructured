## Lab 7 – Dependency Parsing với spaCy

### 1. Mục tiêu

- Làm quen với **cú pháp phụ thuộc** (dependency parsing) trong NLP.
- Biết cách:
  - Cài đặt và sử dụng spaCy với mô hình `en_core_web_md`.
  - Trực quan hóa cây phụ thuộc.
  - Truy cập thông tin `token.dep_`, `token.head`, `token.children`.
  - Duyệt cây để trích xuất thông tin (subject–verb–object, cụm danh từ, đường đi tới ROOT).

### 2. Môi trường & Cài đặt

- Thư viện chính: `spacy`.
- Mô hình sử dụng: `en_core_web_md` (model tiếng Anh kích thước trung bình, có word vectors và thông tin cú pháp).

Lệnh cài đặt trong notebook:

```bash
!pip install -U spacy
!python -m spacy download en_core_web_md
```

Sau đó:

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_md")
```

### 3. Phân tích câu & trực quan hóa

**Ví dụ câu:**  
`"The quick brown fox jumps over the lazy dog."`

**Các bước:**

- Gọi `doc = nlp(text)` để phân tích.
- Dùng `displacy.render(doc, style="dep", jupyter=True, options=...)` để vẽ cây phụ thuộc trong notebook.
- Có thể dùng `displacy.serve(doc, style="dep")` để phục vụ qua HTTP trong môi trường phù hợp.

**Nhận xét từ cây phụ thuộc:**

- ROOT của câu là `"jumps"` (động từ chính).
- `"fox"` là chủ ngữ (`nsubj`) của `"jumps"`.
- `"over"` là giới từ (`prep`) phụ thuộc vào `"jumps"`.
- `"dog"` là tân ngữ giới từ (`pobj`) của `"over"`, kèm các thành phần bổ nghĩa:
  - `"the"` – `det`
  - `"lazy"` – `amod`
- `"fox"` là head của `"The"` (`det`), `"quick"` (`amod`), `"brown"` (`amod`).

### 4. Truy cập thông tin dependency

**Ví dụ câu:**  
`"Apple is looking at buying U.K. startup for $1 billion"`

**Mục tiêu:** In ra thông tin cho mỗi token:

- `TEXT`, `DEP_`, `HEAD TEXT`, `HEAD POS_`, và danh sách `children`.

**Ý nghĩa:**

- Giúp hiểu cấu trúc nội bộ của câu:
  - `"looking"` là ROOT, có children `["Apple", "is", "at"]`.
  - `"buying"` là `pcomp` (complement của giới từ `"at"`).
  - `"startup"` là `dobj` của `"buying"`; `"U.K."` là `compound` gắn với `"startup"`.
  - Cụm `$1 billion` được gắn với `"billion"` (`quantmod`, `compound`), và `"billion"` là `pobj` của `"for"`.

### 5. Duyệt cây để trích xuất thông tin

#### 5.1 Trích xuất (Subject, Verb, Object)

**Ví dụ câu:**  
`"The cat chased the mouse and the dog watched them."`

**Ý tưởng:**

- Duyệt qua các token, chọn những token có `pos_ == "VERB"`.
- Với mỗi động từ:
  - Tìm con (`children`) có `dep_ == "nsubj"` → chủ ngữ.
  - Tìm con có `dep_ == "dobj"` → tân ngữ trực tiếp.
- Nếu tìm được cả subject và object → in bộ ba `(subject, verb, object)`.

**Kết quả:**

- `("cat", "chased", "mouse")`
- `("dog", "watched", "them")`

#### 5.2 Trích xuất các tính từ bổ nghĩa cho danh từ

**Ví dụ câu:**  
`"The big, fluffy white cat is sleeping on the warm mat."`

**Ý tưởng:**

- Duyệt các token, chọn những token có `pos_ == "NOUN"`.
- Với mỗi danh từ:
  - Duyệt các `children`, lấy những token có `dep_ == "amod"` (adjectival modifier).

**Kết quả:**

- `"cat"` được bổ nghĩa bởi `["big", "fluffy", "white"]`.
- `"mat"` được bổ nghĩa bởi `["warm"]`.

#### 5.3 Tìm đường đi ngắn nhất từ một token đến ROOT

**Hàm `get_path_to_root(token)`**:

- Bắt đầu từ `token` hiện tại.
- Lần lượt đi lên `token.head` cho đến khi:
  - `token.head == token` (hoặc `token.dep_ == "ROOT"`).
- Ghi lại chuỗi token trên đường đi.

**Ví dụ:**  
Với token `"brown"` trong câu `"The quick brown fox jumps over the lazy dog."`:

- Đường đi: `brown -> fox -> jumps`
- Cho thấy `"jumps"` là gốc cú pháp, `"fox"` là head trung gian của `"brown"`.

### 6. Bài tập tự luyện

#### Bài 1 – Tìm động từ chính của câu

**Hàm:** `find_main_verb(doc)`

**Yêu cầu:**

- Viết hàm nhận vào một đối tượng `doc` (kết quả của `nlp(text)`).
- Duyệt qua các token trong `doc` để tìm token có:
  - `dep_ == "ROOT"` và `pos_ == "VERB"` → đây thường là **động từ chính** của câu.
  - Nếu ROOT không phải động từ (ví dụ câu đặc biệt, câu danh từ…), vẫn trả về token ROOT đó.

#### Bài 2 – Trích xuất các cụm danh từ (Noun Chunks)

**Hàm:** `extract_noun_chunks_manual(doc)`

**Yêu cầu:**

- Tự hiện thực logic trích xuất **cụm danh từ** mà không dùng sẵn `doc.noun_chunks`.
- Ý tưởng:
  - Duyệt qua các token, chọn các token có `pos_ == "NOUN"`.
  - Với mỗi danh từ:
    - Lấy các token con bên trái (left children) có `dep_` thuộc nhóm mô tả/bổ nghĩa danh từ, ví dụ:
      - `det` (determiner – mạo từ).
      - `amod` (adjectival modifier – tính từ bổ nghĩa).
      - `compound` (danh từ ghép).
    - Ghép các token này + chính danh từ lại thành một chuỗi hoàn chỉnh.
- So sánh kết quả thủ công với `doc.noun_chunks` của spaCy:
  - In ra:
    - `Manual Extraction: [...]`
    - `Spacy Built-in: [...]`

#### Bài 3 – Tìm đường đi ngắn nhất trong cây (Path to ROOT)

**Hàm:** `get_path_to_root(token)`

**Yêu cầu:**

- Viết hàm nhận vào một `token` và trả về danh sách token trên đường đi từ token đó tới gốc cây (ROOT).
- Thuật toán:
  - Khởi tạo `path = [token]`.
  - Trong khi `current_token.head != current_token`:
    - Gán `current_token = current_token.head`.
    - Thêm `current_token` vào `path`.
  - Trả về `path`.

**Ví dụ sử dụng:**

- Lấy token `"brown"` trong câu `"The quick brown fox jumps over the lazy dog."`:
  - Kết quả in ra: `brown -> fox -> jumps`.

- Bài tập này giải thích:
  - Cách đi **ngược** trên cây phụ thuộc bằng thuộc tính `token.head`.
  - Hiểu khái niệm “khoảng cách cú pháp” giữa hai từ: số bước trên đường đi trong cây.
- Đây là tiền đề cho các ứng dụng:
  - Tìm đường đi ngắn nhất giữa hai từ bất kỳ trong câu.
  - Phân tích quan hệ sâu hơn giữa các thành phần xa nhau trên bề mặt câu.

### 7. Nhận xét & Kết luận

- Dependency parsing cung cấp một cấu trúc cây giàu thông tin về:
  - Quan hệ giữa các từ (chủ ngữ, tân ngữ, bổ ngữ, giới từ…).
  - Cấu trúc ngữ pháp bề mặt của câu.
- Việc duyệt cây phụ thuộc cho phép:
  - Trích xuất các bộ ba (Subject, Verb, Object).
  - Tìm các cụm danh từ, cụm động từ.
  - Tính đường đi trong cây để đo “khoảng cách cú pháp” giữa các từ.



