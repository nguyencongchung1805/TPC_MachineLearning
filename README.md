# Dự đoán hàm lượng TPC bằng Machine Learning

## Giới thiệu

Dự án xây dựng và đánh giá các mô hình học máy để dự đoán **Total Phenolic Content (TPC)** trong quá trình chiết xuất từ lá chùm ngây. Giá trị TPC được biểu diễn bằng `mg GAE/g`.

Các điều kiện chiết xuất dùng làm biến đầu vào gồm:

- Nồng độ ethanol — `EtOH (%)`
- Tỷ lệ dung môi/nguyên liệu — `Ratio (mL/g)`
- Nhiệt độ — `Temperature (°C)`
- Thời gian — `Time (min)`

## Mục tiêu

- So sánh hiệu năng của nhiều thuật toán hồi quy.
- Chọn mô hình có khả năng tổng quát hóa tốt bằng kiểm định chéo Leave-One-Out (LOOCV).
- Dự đoán TPC cho các điều kiện chiết xuất mới.

## Kết quả chính

Mô hình tốt nhất là **Gaussian Process Regression (GPR)** với kernel RBF.

| Chỉ số LOOCV | Giá trị |
| --- | ---: |
| R² | 0.9145 |
| MAE | 0.5803 mg GAE/g |
| RMSE | 0.6609 mg GAE/g |

Kết quả chi tiết của các mô hình và quá trình tinh chỉnh được lưu trong thư mục `results/`.

## Cấu trúc thư mục

```text
TPC_MachineLearning/
├── data/
│   ├── tpc_dataset.csv          # Dữ liệu gốc
│   ├── tpc_dataset.xlsx         # Dữ liệu gốc (Excel)
│   ├── X_scaled.csv             # Biến đầu vào đã chuẩn hóa
│   └── y.csv                    # Biến mục tiêu
├── models/
│   ├── best_gpr_model.pkl       # Mô hình GPR tốt nhất
│   ├── best_xgb_model.pkl       # Mô hình XGBoost
│   ├── rsm_model.pkl            # Mô hình RSM
│   └── scaler.pkl               # Bộ chuẩn hóa dữ liệu
├── notebooks/
│   └── TPC_MC.ipynb             # Notebook phân tích, huấn luyện mô hình
├── results/
│   ├── evaluate_all_models.py   # Đánh giá các mô hình
│   ├── gpr_tuning.py            # Tinh chỉnh GPR
│   └── *.csv                    # Kết quả đánh giá/tinh chỉnh
├── run_best_model.py             # Dự đoán bằng mô hình GPR tốt nhất
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. Cài các thư viện cần thiết:

```bash
pip install numpy pandas scikit-learn joblib openpyxl xgboost python-docx
```

## Cách chạy

### 1. Mở notebook để xem quy trình phân tích và huấn luyện

```bash
jupyter notebook notebooks/TPC_MC.ipynb
```

### 2. Dự đoán TPC cho điều kiện mới

Chỉnh các giá trị trong biến `new_samples` của file `run_best_model.py`, sau đó chạy:

```bash
python run_best_model.py
```

Chương trình trả về giá trị TPC dự đoán và độ không chắc chắn (độ lệch chuẩn) của mỗi mẫu.

### 3. Đánh giá các mô hình

```bash
python results/evaluate_all_models.py
```

## Các mô hình được khảo sát

- Response Surface Methodology (RSM) với đa thức bậc 2
- Ridge, Lasso và ElasticNet
- Support Vector Regression (SVR)
- Gaussian Process Regression (GPR)
- Partial Least Squares (PLS)
- Random Forest, Extra Trees, Gradient Boosting và AdaBoost
- XGBoost

## Ghi chú

Dữ liệu và mô hình trong repository phục vụ mục đích học tập/nghiên cứu. Khi áp dụng thực tế, cần kiểm tra lại phạm vi của dữ liệu và xác thực mô hình bằng các thí nghiệm độc lập.
