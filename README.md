# Dữ liệu bài tập lớn xử lý ảnh

## Định dạng
labeled/train/img/...jpg
labeled/train/label.csv

labeled/valid/img/...jpg
labeled/valid/label.csv

labeled/test/img/...jpg
labeled/test/label.csv

### Định dạng csv (name, age, gender)
gender = 0: Nam
gender = 1: Nữ

## Mô tả
age: giá trị liên tục, mean = 33, min = 1, max = 116
gender: giá trị rời rạc (nhị phân)

## Trực quan

### Tỷ lệ chia
![Bar chart Tỷ lệ chia tập dữ liệu](data_visualization/split_ratio.png)

### Phân bố giới tính
![Bar chart giới tính](data_visualization/gender.png)

### Phân bố tuổi
![Histogram tuổi](data_visualization/age.png)