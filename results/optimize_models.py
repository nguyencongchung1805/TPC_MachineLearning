import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "tpc_dataset.xlsx")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load models
gpr_model = joblib.load(os.path.join(MODEL_DIR, "best_gpr_model.pkl"))
rsm_model = joblib.load(os.path.join(MODEL_DIR, "rsm_model.pkl"))
df_raw = pd.read_excel(DATA_PATH)

# Feature names and bounds in physical units
feature_names = ["EtOH (%)", "Ratio (mL/g)", "Temperature (°C)", "Time (min)"]
bounds_real = [
    (60.0, 90.0),   # EtOH (%)
    (16.0, 24.0),   # Ratio (mL/g)
    (45.0, 75.0),   # Temp (°C)
    (60.0, 150.0)   # Time (min)
]

center = np.array([75.0, 20.0, 60.0, 105.0])
half_range = np.array([15.0, 4.0, 15.0, 45.0])

def real_to_coded(x_real):
    return (np.array(x_real) - center) / half_range

def coded_to_real(x_coded):
    return center + np.array(x_coded) * half_range

# ==========================================================
# 1. RSM Objective Function (Phương trình đa thức bậc 2)
# ==========================================================
def rsm_objective_coded(x_coded):
    """
    Hàm mục tiêu RSM dạng mã hóa A, B, C, D trong [-1, 1]
    TPC = 44.733 + 0.600A - 0.250B + 0.483C - 1.100D
          - 4.296A² - 2.396B² - 1.896C² - 1.621D²
          - 0.500AB - 0.100AC - 0.825BC + 0.825BD + 0.225CD
    """
    A, B, C, D = x_coded
    tpc = (44.7333 + 0.600*A - 0.250*B + 0.4833*C - 1.100*D
           - 4.2958*(A**2) - 2.3958*(B**2) - 1.8958*(C**2) - 1.6208*(D**2)
           - 0.500*A*B - 0.100*A*C - 0.825*B*C + 0.825*B*D + 0.225*C*D)
    return tpc

def rsm_neg_objective_real(x_real):
    x_coded = real_to_coded(x_real)
    return -rsm_objective_coded(x_coded)

# ==========================================================
# 2. GPR Objective Function (Mô hình Học máy Gaussian Process)
# ==========================================================
def gpr_neg_objective_real(x_real):
    df_single = pd.DataFrame([x_real], columns=feature_names)
    pred_mean = gpr_model.predict(df_single)[0]
    return -pred_mean

# ==========================================================
# 3. Optimization Execution
# ==========================================================
print("=== RUNNING OPTIMIZATION FOR OBJECTIVE FUNCTIONS ===")

# --- A. RSM Optimization ---
bounds_coded = [(-1.0, 1.0)] * 4
res_rsm_de = differential_evolution(
    lambda x: -rsm_objective_coded(x),
    bounds_coded,
    seed=42
)

best_rsm_coded = res_rsm_de.x
best_rsm_real = coded_to_real(best_rsm_coded)
best_rsm_tpc = -res_rsm_de.fun

# --- B. GPR Optimization ---
res_gpr_de = differential_evolution(
    gpr_neg_objective_real,
    bounds_real,
    seed=42
)

best_gpr_real = res_gpr_de.x
best_gpr_coded = real_to_coded(best_gpr_real)

df_best_gpr = pd.DataFrame([best_gpr_real], columns=feature_names)
best_gpr_tpc, best_gpr_std = gpr_model.predict(df_best_gpr, return_std=True)
best_gpr_tpc = best_gpr_tpc[0]
best_gpr_std = best_gpr_std[0]

# --- C. Highest Observed Experimental Point ---
best_exp_idx = df_raw["TPC (mg GAE/g)"].idxmax()
best_exp_row = df_raw.loc[best_exp_idx]
best_exp_real = best_exp_row[feature_names].values
best_exp_tpc = best_exp_row["TPC (mg GAE/g)"]

# ==========================================================
# 4. Summary Results
# ==========================================================
opt_summary = pd.DataFrame([
    {
        "Method": "RSM (2nd-Order Polynomial)",
        "EtOH (%)": round(best_rsm_real[0], 2),
        "Ratio (mL/g)": round(best_rsm_real[1], 2),
        "Temp (°C)": round(best_rsm_real[2], 2),
        "Time (min)": round(best_rsm_real[3], 2),
        "Coded (A,B,C,D)": f"[{best_rsm_coded[0]:.2f}, {best_rsm_coded[1]:.2f}, {best_rsm_coded[2]:.2f}, {best_rsm_coded[3]:.2f}]",
        "Predicted TPC (mg GAE/g)": round(best_rsm_tpc, 2),
        "Uncertainty (std)": "N/A"
    },
    {
        "Method": "GPR (Gaussian Process Regression)",
        "EtOH (%)": round(best_gpr_real[0], 2),
        "Ratio (mL/g)": round(best_gpr_real[1], 2),
        "Temp (°C)": round(best_gpr_real[2], 2),
        "Time (min)": round(best_gpr_real[3], 2),
        "Coded (A,B,C,D)": f"[{best_gpr_coded[0]:.2f}, {best_gpr_coded[1]:.2f}, {best_gpr_coded[2]:.2f}, {best_gpr_coded[3]:.2f}]",
        "Predicted TPC (mg GAE/g)": round(best_gpr_tpc, 2),
        "Uncertainty (std)": round(best_gpr_std, 4)
    },
    {
        "Method": "Highest Experimental Observed",
        "EtOH (%)": round(best_exp_real[0], 2),
        "Ratio (mL/g)": round(best_exp_real[1], 2),
        "Temp (°C)": round(best_exp_real[2], 2),
        "Time (min)": round(best_exp_real[3], 2),
        "Coded (A,B,C,D)": "[0.00, 0.00, 0.00, 0.00]",
        "Predicted TPC (mg GAE/g)": round(best_exp_tpc, 2),
        "Uncertainty (std)": "0.00 (Experimental)"
    }
])

print("\n" + opt_summary.to_string(index=False))

# Save output
csv_save_path = os.path.join(RESULTS_DIR, "optimization_results.csv")
opt_summary.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
print(f"\nSaved optimization summary to: {csv_save_path}")

