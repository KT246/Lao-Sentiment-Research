# Ghi chú quy trình thu thập, tiền xử lý và huấn luyện dữ liệu

File này được ghi lại sau khi đọc tham khảo hai notebook:
- `D:\AI\2026\tien-xu-ly\thu_thap_data.ipynb`
- `D:\AI\2026\tien-xu-ly\trian_tien_xu_ly.ipynb`

Mục đích của tài liệu là tóm tắt lại đúng mạch công việc nhóm đã thực hiện, theo trình tự:

`thu thập dữ liệu -> tiền xử lý -> train/cross-validation để lọc bớt câu lỗi -> gộp bộ dữ liệu sạch -> chia train/validation -> train benchmark -> lấy kết quả`

## 1. Tổng quan

Hai notebook cho thấy nhóm không đi theo hướng thu thập xong rồi chia train/validation ngay. Thay vào đó, dữ liệu được xử lý qua nhiều lớp để giảm nhiễu trước khi dùng cho benchmark cuối.

Mạch thực tế gồm:
1. Thu thập review từ Google Play
2. Gán nhãn ban đầu từ số sao
3. Làm sạch văn bản và loại dữ liệu rác
4. Dùng train/cross-validation để phát hiện các câu dễ sai hoặc gây tranh cãi
5. Giữ lại phần dữ liệu ổn định hơn
6. Chia train/validation
7. Huấn luyện benchmark và tổng hợp kết quả

## 2. Giai đoạn thu thập dữ liệu

Notebook `thu_thap_data.ipynb` sử dụng `google_play_scraper` để lấy review từ Google Play.

### 2.1 Nguồn ứng dụng

Review được lấy từ nhiều nhóm ứng dụng phục vụ người dùng ở Lào, gồm:
- ngân hàng và tài chính
- viễn thông
- giao thông
- dịch vụ chính phủ
- thương mại và dịch vụ số
- một số nền tảng xã hội như TikTok, Facebook, Messenger, WhatsApp

### 2.2 Cách quét

Notebook dùng nhiều chiến lược để mở rộng độ phủ:
- quét cả store `la` và `th`
- quét theo ngôn ngữ hệ thống `lo`, `th`, `en`
- với một số app xã hội, có phiên bản chỉ lấy review chứa tiếng Lào
- với app nhỏ, cố gắng lấy toàn bộ lịch sử
- với app lớn, có phiên bản chỉ lấy dữ liệu từ một mốc năm trở đi để tránh review quá cũ

### 2.3 Lọc tiếng Lào và gán nhãn ban đầu

Ngay sau khi quét, notebook:
- chỉ giữ lại các review có tín hiệu ký tự Lào
- xử lý trùng ở mức `reviewId` trong lúc thu thập để tránh lặp bản ghi
- gán nhãn ban đầu từ số sao:
  - `1-2 sao` -> Negative
  - `3 sao` -> Neutral
  - `4-5 sao` -> Positive

Đây mới là lớp nhãn ban đầu, chưa phải bộ dữ liệu cuối dùng cho benchmark.

## 3. Giai đoạn tiền xử lý

Notebook `trian_tien_xu_ly.ipynb` tiếp tục xử lý dữ liệu sau bước thu thập.

### 3.1 Đưa bài toán về nhị phân

Notebook thực hiện các bước:
- ép cột `score` về dạng số
- loại bỏ các dòng lỗi định dạng
- bỏ toàn bộ review `3 sao`
- chuyển nhãn thành:
  - `0` = Negative
  - `1` = Positive
- đổi tên cột `content` thành `text`

### 3.2 Làm sạch văn bản

Phần làm sạch gồm:
- bỏ emoji
- chuẩn hóa Unicode bằng `NFKC`
- chuyển về chữ thường
- xóa URL và email
- chuẩn hóa một số lỗi ký tự Lào
- giảm lặp ký tự và dấu câu
- chỉ giữ lại chủ yếu:
  - tiếng Lào
  - tiếng Anh
  - chữ số
  - một phần ký tự Thái
  - dấu câu cơ bản

### 3.3 Lọc dữ liệu rác

Notebook dùng nhiều luật để loại các câu kém chất lượng, chẳng hạn:
- quá ít ký tự có nghĩa
- không đủ tín hiệu tiếng Lào
- tiếng Thái lấn át tiếng Lào
- tiếng Anh xuất hiện quá dày
- cấu trúc nguyên âm và phụ âm bất thường
- dấu Lào đi sai với số hoặc chữ Latin
- chuỗi lặp bất thường
- mật độ ký tự méo
- quá nhiều token rất ngắn

Sau bước này, notebook tạo ra:
- một file dữ liệu sạch
- một file log để kiểm tra lại phần đã bị loại

## 4. Dùng train/cross-validation để lọc dữ liệu

Đây là phần quan trọng nhất của quy trình. Nhóm mình không dừng ở làm sạch theo luật, mà còn dùng mô hình để hỗ trợ sàng lọc dữ liệu.

### 4.1 Chạy cross-validation

Notebook chạy `5-fold cross-validation` với mô hình như `xlm-roberta-base`.

Trong bước này:
- dữ liệu được chia bằng `StratifiedKFold`
- có tính `class weights`
- dùng `CrossEntropyLoss` có trọng số
- lưu dự đoán của từng fold

### 4.2 Tìm câu dễ sai

Sau khi có kết quả từ 5 fold, notebook:
- gộp toàn bộ file dự đoán
- lọc ra các câu có:
  - `true_label != predicted_label`

Những câu này được xem là:
- khó phân loại
- dễ gây tranh cãi
- hoặc có khả năng gán nhãn chưa ổn

### 4.3 Tạo bộ dữ liệu vàng

Từ kết quả trên, notebook giữ lại phần dữ liệu ổn định hơn để tạo `golden dataset`.

Ý tưởng chính là:
- giảm bớt vùng xám
- loại các câu dễ gây sai lệch
- giữ lại những mẫu đáng tin cậy hơn cho thí nghiệm cuối

Nói ngắn gọn, bước train ở giai đoạn này dùng để hỗ trợ lọc dữ liệu, không chỉ để xem điểm mô hình.

## 5. Gộp thành bộ dữ liệu sạch

Sau các bước trên, dữ liệu được đưa về một bộ sạch hơn:
- đã bỏ neutral
- đã làm sạch văn bản
- đã loại phần rác rõ ràng
- đã giảm bớt các câu dễ sai thông qua cross-validation

Ở đây trọng tâm là làm sạch và tinh lọc, không mô tả pipeline như một bước `dedup text` riêng biệt.

Kết quả của giai đoạn này là bộ dữ liệu cuối, thường được notebook gọi là:
- `golden dataset`
- hoặc bộ dữ liệu sạch để chia train/validation

## 6. Chia train/validation

Sau khi có bộ dữ liệu cuối, notebook mới thực hiện chia dữ liệu thành:
- `train`
- `validation`

Cách chia:
- tỷ lệ `80/20`
- có `stratify` theo nhãn để giữ phân bố lớp

Đây là bộ split dùng cho huấn luyện và đánh giá benchmark chính.

## 7. Sau khi chia dữ liệu xong thì làm gì tiếp

Sau khi tạo `train.csv` và `val.csv`, pipeline tiếp tục như sau:

### 7.1 Nạp dữ liệu

Notebook đọc:
- `train.csv`
- `val.csv`

Sau đó chuyển dữ liệu sang định dạng phù hợp để huấn luyện.

### 7.2 Tokenize và chuẩn bị mô hình

Notebook:
- load tokenizer từ checkpoint
- tokenize cột `text`
- load mô hình phân loại với `num_labels=2`

Các checkpoint xuất hiện trong notebook gồm:
- `w11wo/lao-roberta-base`
- `xlm-roberta-base`
- `bert-base-multilingual-cased`

### 7.3 Huấn luyện benchmark

Từ bộ split cuối, notebook bắt đầu fine-tuning và benchmark:
- train trên tập train
- evaluate trên tập validation
- lưu model vào thư mục output riêng

Ngoài bản train cơ bản, notebook còn có các phiên bản mở rộng để:
- log thời gian từng epoch
- ghi nhận phần cứng
- xuất file dự đoán chi tiết

### 7.4 Xuất file kết quả

Sau mỗi lần train, notebook thường lưu:
- model đã huấn luyện
- tokenizer
- file dự đoán chi tiết trên validation
- confidence score
- file tổng hợp benchmark

Những file này được dùng để:
- lập bảng kết quả
- soi lỗi
- phân tích false positive và false negative
- viết báo cáo

## 8. Kết quả và các số liệu quan trọng từ output notebook

Từ `trian_tien_xu_ly.ipynb`, có thể ghi lại các mốc dữ liệu sau:

### 8.1 Sau bước gán nhãn nhị phân

- Tổng số dòng ban đầu: `134,058`
- Loại bỏ `2` dòng lỗi định dạng `score`
- Sau khi bỏ review `3 sao`: `125,597`

Phân bố nhãn ở giai đoạn này:
- Label `0` (Negative): `29,862`
- Label `1` (Positive): `95,735`

### 8.2 Sau bước làm sạch và lọc rác

- Số câu rác bị loại: `32,176`
- Dữ liệu sạch còn lại: `31,769`

### 8.3 Sau bước cross-validation để hỗ trợ lọc

- dùng `5-fold cross-validation`
- class weights: `[1.3704167, 0.78721875]`
- tổng số câu ở giai đoạn CV: `31,769`
- số câu AI đoán sai: `6,630`

### 8.4 Sau bước tạo bộ dữ liệu vàng

- Số câu bị loại khỏi bộ vàng: `6,630`
- Bộ dữ liệu vàng còn lại: `25,139`

### 8.5 Sau bước chia train/validation

Từ `25,139` câu của bộ dữ liệu vàng, notebook chia:
- Train: `20,111`
- Validation: `5,028`

Đây chính là bộ split cuối đang được dùng trong paper.

## 9. Các file đầu ra đáng chú ý

### 9.1 Giai đoạn thu thập
- `lao_script_data_SMART_FILTER.csv`
- `lao_script_data_FINAL_STRATEGY.csv`

### 9.2 Giai đoạn gán nhãn và làm sạch
- `Lao_Dataset_Labeled_Raw.csv`
- `Lao_Dataset_For_LaoPLM_Binary.csv`
- `Garbage_Logs_V8.csv`

### 9.3 Giai đoạn lọc bằng train/CV
- `cv1_results.csv` ... `cv5_results.csv`
- `CrossVal_Summary_Score.csv`
- `Danh_Sach_Cau_Sai_Can_Sua.xlsx`
- `Lao_Dataset_GOLDEN.csv`

### 9.4 Giai đoạn chia và train
- `train.csv`
- `val.csv`
- thư mục model output
- file dự đoán chi tiết trên validation
- file benchmark summary

## 10. Tóm tắt ngắn gọn

Nếu cần mô tả ngắn nhất theo đúng thực tế, có thể viết như sau:

> Nhóm thu thập review từ Google Play, giữ lại các review có chữ Lào, gán nhãn ban đầu từ số sao, làm sạch văn bản và loại dữ liệu rác, sau đó dùng cross-validation để phát hiện các câu dễ sai hoặc gây tranh cãi nhằm tạo bộ dữ liệu sạch hơn. Từ bộ dữ liệu đã tinh lọc, nhóm chia train/validation, tiếp tục train benchmark các mô hình, lấy predictions và metrics, rồi dùng các kết quả đó cho phân tích lỗi và viết báo cáo.
