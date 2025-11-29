# ✈️ Phân Loại Mức Độ Hài Lòng Của Khách Hàng Hàng Không (Airline Customer Satisfaction Classification)

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/Libraries-Pandas%2C%20Sklearn%2C%20Seaborn%2C%20Matplotlib-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 1. Tổng Quan Dự Án

Dự án này nhằm mục đích xây dựng một mô hình Machine Learning để **phân loại** mức độ hài lòng của khách hàng đối với dịch vụ hàng không dựa trên các yếu tố như dịch vụ bay, tiện nghi, thông tin chuyến bay, và các yếu tố nhân khẩu học. Mục tiêu cuối cùng là giúp hãng hàng không xác định các yếu tố quan trọng nhất ảnh hưởng đến sự hài lòng, từ đó cải thiện chất lượng dịch vụ.

* **Mục tiêu chính:** Dự đoán kết quả đầu ra là **Satisfied (Hài lòng)** hoặc **Neutral or Dissatisfied (Trung lập/Không hài lòng)**.
* **Vấn đề nghiệp vụ:** Giảm thiểu tỷ lệ khách hàng không hài lòng và tăng cường trải nghiệm bay.

## 💾 2. Nguồn Dữ Liệu

| Thông tin | Chi tiết |
| :--- | :--- |
| **Nguồn gốc** | [Kaggle - Airline Passenger Satisfaction] |
| **Kích thước** | 26000 hàng và 23 cột. |
| **Biến mục tiêu** | `satisfaction` (Hài lòng/Không hài lòng). |
| **Các đặc trưng chính** | `Type of Travel`, `Class`, `Inflight wifi service`, `Cleanliness`, `On-board service`, `Gender`, `Customer Type`, v.v. |

## 🧪 3. Phương Pháp Luận và Mô Hình

### 3.1. Tiền xử lý Dữ liệu (Preprocessing)

* **Xử lý thiếu dữ liệu (Missing Data):** [Mô tả cách xử lý, ví dụ: Điền giá trị trung bình/mode hoặc loại bỏ.]
* **Mã hóa dữ liệu phân loại (Encoding):** [Mô tả kỹ thuật, ví dụ: One-Hot Encoding cho các biến nominal.]
* **Chuẩn hóa/Thay đổi tỷ lệ (Scaling):** [Mô tả kỹ thuật, ví dụ: StandardScaler cho các biến numeric.]

### 3.2. Mô hình Đã Thử Nghiệm

Các mô hình sau đã được thử nghiệm và đánh giá:

* Logistic Regression
* [Tên mô hình tốt nhất, ví dụ: **XGBoost Classifier**]

## 📊 4. Kết Quả và Đánh Giá

Mô hình tốt nhất được chọn là **[Tên mô hình tốt nhất]** dựa trên chỉ số **[Tên chỉ số chính, ví dụ: F1-Score]** trên tập dữ liệu thử nghiệm (Test Set).

| Mô hình | Accuracy (%) | Precision (Satisfied) | Recall (Satisfied) | F1-Score (Satisfied) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | [XX.XX]% | [XX.XX]% | [XX.XX]% | [XX.XX]% |

* **Nhận xét chính:** [Ví dụ: Mô hình XGBoost cho thấy sự cân bằng tốt nhất giữa Precision và Recall.]

### 4.1. Tầm quan trọng của Đặc trưng (Feature Importance)

Các đặc trưng quan trọng nhất trong việc dự đoán mức độ hài lòng là:
1.  `Inflight wifi service`
2.  `Ease of Online booking`
3.  `Type of Travel`
4.  [Đặc trưng khác 4]

## 💻 5. Cài Đặt và Chạy Dự Án

Để tái tạo dự án này trên máy tính của bạn, hãy làm theo các bước sau:

### 5.1. Yêu cầu Hệ thống

* Python [Phiên bản 3.13 trở lên]

### 5.2. Cài đặt Thư viện

Tạo môi trường ảo và cài đặt các thư viện cần thiết:

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
venv\Scripts\activate     # Trên Windows

# Cài đặt các thư viện từ file requirements.txt
pip install -r requirements.txt
