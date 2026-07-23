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

# Load models
gpr_path = os.path.join(MODEL_DIR, "best_gpr_model.pkl")
rsm_path = os.path.join(MODEL_DIR, "rsm_model.pkl")

if not os.path.exists(gpr_path):
    raise FileNotFoundError(f"Model GPR not found at {gpr_path}")
if not os.path.exists(rsm_path):
    raise FileNotFoundError(f"Model RSM not found at {rsm_path}")

gpr_model = joblib.load(gpr_path)
rsm_model = joblib.load(rsm_path)
print("Loaded models successfully.")

# Setup default center values for parameters
defaults = {
    "EtOH (%)": 75.0,
    "Ratio (mL/g)": 20.0,
    "Temperature (°C)": 60.0,
    "Time (min)": 105.0
}

def plot_surface_contour(var1_name, var1_range, var2_name, var2_range, filename_suffix, title_suffix):
    """
    Plots GPR and RSM 3D response surfaces and 2D contour plots side by side.
    """
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
    
    # Predict TPC
    z_gpr = gpr_model.predict(df_pred)
    z_rsm = rsm_model.predict(df_pred)
    
    Z_gpr = z_gpr.reshape(X.shape)
    Z_rsm = z_rsm.reshape(X.shape)
    
    # Scale colorbar consistently between GPR and RSM
    z_min = min(z_gpr.min(), z_rsm.min())
    z_max = max(z_gpr.max(), z_rsm.max())
    
    fig = plt.figure(figsize=(16, 12))
    
    # ------------------ GPR Plots ------------------
    # 1. GPR 3D Surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_gpr, cmap='magma', edgecolor='none', alpha=0.9, vmin=z_min, vmax=z_max)
    ax1.set_xlabel(var1_name, fontsize=10, labelpad=8)
    ax1.set_ylabel(var2_name, fontsize=10, labelpad=8)
    ax1.set_zlabel("TPC (mg GAE/g)", fontsize=10, labelpad=8)
    ax1.set_title(f"A. Bề mặt đáp ứng GPR (Học máy)\n{title_suffix}", fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=30, azim=-60)
    
    # 2. GPR 2D Contour
    ax2 = fig.add_subplot(2, 2, 2)
    contour1 = ax2.contourf(X, Y, Z_gpr, levels=15, cmap='magma', vmin=z_min, vmax=z_max)
    lines1 = ax2.contour(X, Y, Z_gpr, levels=10, colors='white', linewidths=0.5)
    ax2.clabel(lines1, inline=True, fontsize=8, fmt='%.1f')
    ax2.set_xlabel(var1_name, fontsize=10)
    ax2.set_ylabel(var2_name, fontsize=10)
    ax2.set_title("B. Đường đồng mức GPR\n", fontsize=12, fontweight='bold')
    
    # Mark local maximum
    idx_max_gpr = np.argmax(z_gpr)
    x_max_gpr = df_pred[var1_name].iloc[idx_max_gpr]
    y_max_gpr = df_pred[var2_name].iloc[idx_max_gpr]
    ax2.plot(x_max_gpr, y_max_gpr, 'r*', markersize=12, label=f'Cực đại GPR: {z_gpr[idx_max_gpr]:.2f} mg/g')
    ax2.legend(loc='lower left', fontsize=9)
    
    # ------------------ RSM Plots ------------------
    # 3. RSM 3D Surface
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_rsm, cmap='magma', edgecolor='none', alpha=0.9, vmin=z_min, vmax=z_max)
    ax3.set_xlabel(var1_name, fontsize=10, labelpad=8)
    ax3.set_ylabel(var2_name, fontsize=10, labelpad=8)
    ax3.set_zlabel("TPC (mg GAE/g)", fontsize=10, labelpad=8)
    ax3.set_title(f"C. Bề mặt đáp ứng RSM (Đa thức bậc 2)\n{title_suffix}", fontsize=12, fontweight='bold', pad=10)
    ax3.view_init(elev=30, azim=-60)
    
    # 4. RSM 2D Contour
    ax4 = fig.add_subplot(2, 2, 4)
    contour3 = ax4.contourf(X, Y, Z_rsm, levels=15, cmap='magma', vmin=z_min, vmax=z_max)
    lines3 = ax4.contour(X, Y, Z_rsm, levels=10, colors='white', linewidths=0.5)
    ax4.clabel(lines3, inline=True, fontsize=8, fmt='%.1f')
    ax4.set_xlabel(var1_name, fontsize=10)
    ax4.set_ylabel(var2_name, fontsize=10)
    ax4.set_title("D. Đường đồng mức RSM\n", fontsize=12, fontweight='bold')
    
    # Mark local maximum
    idx_max_rsm = np.argmax(z_rsm)
    x_max_rsm = df_pred[var1_name].iloc[idx_max_rsm]
    y_max_rsm = df_pred[var2_name].iloc[idx_max_rsm]
    ax4.plot(x_max_rsm, y_max_rsm, 'r*', markersize=12, label=f'Cực đại RSM: {z_rsm[idx_max_rsm]:.2f} mg/g')
    ax4.legend(loc='lower left', fontsize=9)
    
    # Layout adjustments and colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.22)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(surf1, cax=cbar_ax, label="TPC (mg GAE/g)")
    
    # Save
    save_path = os.path.join(PLOT_DIR, f"comparison_{filename_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot comparison to: {save_path}")

# Run for three key combinations
# 1. EtOH vs Ratio
plot_surface_contour(
    "EtOH (%)", [60, 90],
    "Ratio (mL/g)", [16, 24],
    "etoh_ratio",
    "Cố định: Nhiệt độ = 60°C, Thời gian = 105 phút"
)

# 2. Ratio vs Temperature
plot_surface_contour(
    "Ratio (mL/g)", [16, 24],
    "Temperature (°C)", [45, 75],
    "ratio_temp",
    "Cố định: EtOH = 75%, Thời gian = 105 phút"
)

# 3. EtOH vs Temperature
plot_surface_contour(
    "EtOH (%)", [60, 90],
    "Temperature (°C)", [45, 75],
    "etoh_temp",
    "Cố định: Tỷ lệ = 20 mL/g, Thời gian = 105 phút"
)
