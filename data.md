# Mô tả dữ liệu

Thư mục này chứa mô tả về các dataset được sử dụng trong các bài lab. Các file dữ liệu lớn không được lưu trữ trong repository này để tránh làm quá tải token đầu vào của AI và giúp model tập trung vào nội dung chính.

## Datasets

### 1. UD_English-EWT
- **Mô tả**: Universal Dependencies English EWT (English Web Treebank) - tập dữ liệu dependency parsing và POS tagging cho tiếng Anh
- **Định dạng**: `.conllu` (CoNLL-U format), `.txt`
- **Cấu trúc**:
  - `en_ewt-ud-train.conllu`: Tập huấn luyện
  - `en_ewt-ud-dev.conllu`: Tập validation
  - `en_ewt-ud-test.conllu`: Tập kiểm thử
- **Sử dụng trong**: Lab 1-2, Lab 4, Lab 8
- **Nguồn**: https://universaldependencies.org/treebanks/en_ewt/

### 2. C4 Dataset
- **Mô tả**: Colossal Clean Crawled Corpus - tập dữ liệu văn bản lớn được thu thập từ web
- **Định dạng**: `.json.gz` (JSON Lines, nén gzip)
- **File**: `c4-train.00000-of-01024-30K.json.gz` (30K mẫu từ shard đầu tiên)
- **Cấu trúc**: Mỗi dòng JSON chứa một document với các trường như `text`, `url`, `timestamp`
- **Sử dụng trong**: Lab 2, Lab 4
- **Nguồn**: https://github.com/allenai/c4

### 3. Sentiments Dataset
- **Mô tả**: Tập dữ liệu phân loại cảm xúc (sentiment analysis)
- **Định dạng**: `.csv`
- **File**: `sentiments.csv`
- **Cấu trúc**: Chứa các cột như `text`, `label` (positive/negative)
- **Sử dụng trong**: Lab 5

## Lưu ý

- Các file dữ liệu lớn (>10MB) không được commit vào repository
- Chỉ lưu trữ các file mô tả và metadata
- Để tải dữ liệu đầy đủ, tham khảo các nguồn chính thức được liệt kê ở trên

