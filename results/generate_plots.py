import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set matplotlib parameters for high-quality visuals and support for Vietnamese text
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load best model (GPR)
gpr_path = os.path.join(MODEL_DIR, "best_gpr_model.pkl")
if not os.path.exists(gpr_path):
    raise FileNotFoundError(f"Model GPR not found at {gpr_path}")

gpr_model = joblib.load(gpr_path)
print("Loaded best model (GPR) successfully.")

# Setup default center values for parameters
defaults = {
    "EtOH (%)": 75.0,
    "Ratio (mL/g)": 20.0,
    "Temperature (°C)": 60.0,
    "Time (min)": 105.0
}

def plot_best_model_surface(var1_name, var1_range, var2_name, var2_range, filename_suffix, title_suffix):
    # Create coordinate grid
    x = np.linspace(var1_range[0], var1_range[1], 100)
    y = np.linspace(var2_range[0], var2_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Prepare inputs for prediction
    grid_size = len(x) * len(y)
    df_pred = pd.DataFrame({
        "EtOH (%)": np.full(grid_size, defaults["EtOH (%)"]),
        "Ratio (mL/g)": np.full(grid_size, defaults["Ratio (mL/g)"]),
        "Temperature (°C)": np.full(grid_size, defaults["Temperature (°C)"]),
        "Time (min)": np.full(grid_size, defaults["Time (min)"])
    })
    
    df_pred[var1_name] = X.ravel()
    df_pred[var2_name] = Y.ravel()
    
    # Predict TPC mean and standard deviation (uncertainty)
    z_gpr, z_std = gpr_model.predict(df_pred, return_std=True)
    
    Z_gpr = z_gpr.reshape(X.shape)
    Z_std = z_std.reshape(X.shape)
    
    fig = plt.figure(figsize=(18, 5.5))
    
    # ------------------ 1. 3D Response Surface ------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_gpr, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel(var1_name, fontsize=10, labelpad=8)
    ax1.set_ylabel(var2_name, fontsize=10, labelpad=8)
    ax1.set_zlabel("TPC (mg GAE/g)", fontsize=10, labelpad=8)
    ax1.set_title(f"A. Bề mặt đáp ứng 3D (Mô hình GPR tối ưu)\n{title_suffix}", fontsize=11, fontweight='bold', pad=10)
    ax1.view_init(elev=30, azim=-60)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, label="TPC (mg GAE/g)")
    
    # ------------------ 2. 2D Contour Plot ------------------
    ax2 = fig.add_subplot(1, 2, 2)
    contour1 = ax2.contourf(X, Y, Z_gpr, levels=15, cmap='viridis')
    lines1 = ax2.contour(X, Y, Z_gpr, levels=10, colors='white', linewidths=0.5)
    ax2.clabel(lines1, inline=True, fontsize=8, fmt='%.1f')
    ax2.set_xlabel(var1_name, fontsize=10)
    ax2.set_ylabel(var2_name, fontsize=10)
    ax2.set_title("B. Đường đồng mức Contour TPC\n", fontsize=11, fontweight='bold')
    fig.colorbar(contour1, ax=ax2, label="TPC dự đoán (mg GAE/g)")
    
    # Mark local maximum
    idx_max = np.argmax(z_gpr)
    x_max = df_pred[var1_name].iloc[idx_max]
    y_max = df_pred[var2_name].iloc[idx_max]
    ax2.plot(x_max, y_max, 'r*', markersize=14, label=f'Cực đại: {z_gpr[idx_max]:.2f} mg/g')
    ax2.legend(loc='lower left', fontsize=9)
    
    # Save
    save_path = os.path.join(PLOT_DIR, f"best_model_surface_{filename_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved best model plot to: {save_path}")

# Run for key variable combinations
# 1. EtOH vs Ratio
plot_best_model_surface(
    "EtOH (%)", [60, 90],
    "Ratio (mL/g)", [16, 24],
    "etoh_ratio",
    "Cố định: Nhiệt độ = 60°C, Thời gian = 105 phút"
)

# 2. Ratio vs Temperature
plot_best_model_surface(
    "Ratio (mL/g)", [16, 24],
    "Temperature (°C)", [45, 75],
    "ratio_temp",
    "Cố định: EtOH = 75%, Thời gian = 105 phút"
)

# 3. EtOH vs Temperature
plot_best_model_surface(
    "EtOH (%)", [60, 90],
    "Temperature (°C)", [45, 75],
    "etoh_temp",
    "Cố định: Tỷ lệ = 20 mL/g, Thời gian = 105 phút"
)

# 4. Time vs Temperature
plot_best_model_surface(
    "Time (min)", [75, 150],
    "Temperature (°C)", [45, 75],
    "time_temp",
    "Cố định: EtOH = 75%, Tỷ lệ = 20 mL/g"
)

# 5. Time vs Ratio
plot_best_model_surface(
    "Time (min)", [75, 150],
    "Ratio (mL/g)", [16, 24],
    "time_ratio",
    "Cố định: EtOH = 75%, Nhiệt độ = 60°C"
)

# 6. Time vs EtOH
plot_best_model_surface(
    "Time (min)", [75, 150],
    "EtOH (%)", [60, 90],
    "time_etoh",
    "Cố định: Tỷ lệ = 20 mL/g, Nhiệt độ = 60°C"
)

