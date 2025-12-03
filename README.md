# NLP Labs - Natural Language Processing Projects

Repository này chứa các bài thực hành và dự án về Xử lý Ngôn ngữ Tự nhiên (NLP) được thực hiện trong học kỳ.

## Cấu trúc thư mục

```
./
├── src/              # Chứa các code như .py, .scala, ...
├── report/           # Chứa các báo cáo (lab1_part1.md, lab2_part2.pdf, ...)
├── notebook/         # Chứa các notebook để code nhanh một chủ đề
├── test/             # Chứa code ghi test
├── data/             # Chứa mô tả về dữ liệu (không chứa dataset lớn)
└── README.md         # File này
```

## Danh sách các Lab

### Lab 1-2: Text Tokenization & Count Vectorization
- **Mô tả**: Thực hiện tokenization và vectorization cơ bản
- **Báo cáo**: `report/lab1-2.md`
- **Code**: `src/preprocessing/`, `src/representations/`

### Lab 2: NLP Pipeline với Apache Spark
- **Mô tả**: Xây dựng pipeline xử lý văn bản sử dụng Spark MLlib
- **Báo cáo**: `report/lab2.md`
- **Code**: `src/spark/`

### Lab 4: Word Embeddings với Word2Vec
- **Mô tả**: Triển khai word embeddings với GloVe và Word2Vec
- **Báo cáo**: `report/lab4.md`
- **Code**: `src/representations/word_embedder.py`

### Lab 5: Phân loại Văn bản (Text Classification)
- **Mô tả**: Xây dựng pipeline phân loại văn bản với scikit-learn và PySpark
- **Báo cáo**: `report/lab5.md`
- **Code**: `src/models/text_classifier.py`

### Lab 6: Làm quen với PyTorch
- **Mô tả**: Thực hành cơ bản với Tensor, autograd, và nn.Module
- **Báo cáo**: `report/lab6.md`
- **Notebook**: `notebook/pytorch_intro.ipynb`

### Lab 7: RNNs cho Text Classification
- **Mô tả**: Phân loại intent sử dụng RNN trên dataset HWU
- **Báo cáo**: `report/lab7.md`, `report/lab5_rnns_text_classification.pdf`
- **Notebook**: `notebook/rnns_text_classification.ipynb`

### Lab 8: RNN for POS Tagging
- **Mô tả**: Xây dựng mô hình BiLSTM cho POS tagging
- **Báo cáo**: `report/lab8.md`, `report/lab5_rnn_for_pos_tagging.pdf`
- **Notebook**: `notebook/lab5_rnn_pos_tagging.ipynb`

## Cài đặt và chạy

### Yêu cầu
- Python 3.11+
- Java (cho Spark)
- Scala 2.12+ (cho Spark projects)

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy tests
```bash
# Chạy test cho Lab 1-2
python -m test.main

# Chạy test cho Lab 2
python -m test.lab2_test

# Chạy test cho Lab 4
python test/Lab4_test.py

# Chạy test cho Lab 5
python test/lab5_test.py
```

## Datasets

Các dataset được sử dụng:
- **UD English-EWT**: Universal Dependencies English Web Treebank
- **C4 Dataset**: Colossal Clean Crawled Corpus
- **HWU Dataset**: Home Assistant Understanding dataset
- **Sentiments Dataset**: Tập dữ liệu phân loại cảm xúc

Xem chi tiết trong `data/README.md`.

## Báo cáo

Tất cả các báo cáo được lưu trong thư mục `report/`:

### Báo cáo các Lab
- `lab1-2.md`: Báo cáo Lab 1-2
- `lab2.md`: Báo cáo Lab 2
- `lab4.md`: Báo cáo Lab 4
- `lab5.md`: Báo cáo Lab 5
- `lab6.md`: Báo cáo Lab 6
- `lab7.md`: Báo cáo Lab 7
- `lab8.md`: Báo cáo Lab 8
- `lab5_rnns_text_classification.pdf`: PDF báo cáo Lab 7
- `lab5_rnn_for_pos_tagging.pdf`: PDF báo cáo Lab 8

### Nghiên cứu bổ sung
- `tts_research.md`: Nghiên cứu tổng quan về Text To Speech (TTS) - bao gồm các phương pháp triển khai, ưu nhược điểm, và pipeline tối ưu

## 🔧 Công nghệ sử dụng

- **Python**: scikit-learn, PyTorch, Gensim, PySpark
- **Scala**: Apache Spark MLlib
- **Frameworks**: TensorFlow, PyTorch
- **Tools**: Jupyter Notebook, Apache Spark

## 📄 License

Các dataset và code tuân theo license của từng nguồn tương ứng.

## 👤 Tác giả

Repository này được tạo cho mục đích học tập và nghiên cứu.

