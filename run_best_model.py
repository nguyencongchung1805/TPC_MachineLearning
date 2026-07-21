"""
run_best_model.py
-----------------
Load mô hình GPR tốt nhất và dự đoán TPC từ dữ liệu mới.

Cách dùng:
    python run_best_model.py

Thay đổi giá trị trong `new_samples` để dự đoán cho điều kiện trích xuất mới.
"""

import numpy as np
import pandas as pd
import joblib
import os

# CONFIG
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_gpr_model.pkl")

# LOAD MODEL
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}\n"
                            f"Please run the notebook first to train and save the model.")

gpr_model = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# INPUT DATA
# Thay thế các giá trị dưới đây bằng điều kiện thực nghiệm cần dự đoán
# Cột: EtOH (%), Ratio (mL/g), Temperature (°C), Time (min)
new_samples = pd.DataFrame({
    "EtOH (%)":        [75, 60, 90, 60, 76.15],
    "Ratio (mL/g)":    [20, 16, 24, 20, 19.41],
    "Temperature (°C)":[75, 60, 90, 60, 62],
    "Time (min)":      [105, 60, 150, 90, 88.47],
})

print("\n=== INPUT ===")
print(new_samples.to_string(index=False))

# PREDICT
y_pred, y_std = gpr_model.predict(new_samples, return_std=True)

result = new_samples.copy()
result["Predicted TPC (mg GAE/g)"] = np.round(y_pred, 4)
result["Uncertainty (std)"]         = np.round(y_std, 4)

print("\n=== PREDICTION RESULTS ===")
print(result.to_string(index=False))
# print(f"\nModel LOOCV R² = 0.9145  |  RMSE = 0.6610  |  MAE = 0.5581")
