import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

xlsx_path = r"d:\do an\TPC_MachineLearning\data\tpc_dataset.xlsx"
df = pd.read_excel(xlsx_path)

X = df.drop(columns=["Run", "TPC (mg GAE/g)"])
y = df["TPC (mg GAE/g)"]

loo = LeaveOneOut()

# Define parameter grids
alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
kernels = {
    'RBF': lambda length: C(1.0, (1e-4, 1e4)) * RBF(length, (1e-2, 1e2)),
    'Matern_1.5': lambda length: C(1.0, (1e-4, 1e4)) * Matern(length, (1e-2, 1e2), nu=1.5),
    'Matern_2.5': lambda length: C(1.0, (1e-4, 1e4)) * Matern(length, (1e-2, 1e2), nu=2.5),
    'RationalQuadratic': lambda length: C(1.0, (1e-4, 1e4)) * RationalQuadratic(length_scale=length, alpha=1.0)
}
lengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

results = []

for kernel_name, kernel_fn in kernels.items():
    for length in lengths:
        for alpha in alphas:
            try:
                k = kernel_fn(length)
                gpr_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GaussianProcessRegressor(kernel=k, alpha=alpha, n_restarts_optimizer=15, random_state=42))
                ])
                y_pred = cross_val_predict(gpr_pipeline, X, y, cv=loo)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                
                # Fit on full data to see training score and optimized kernel
                gpr_pipeline.fit(X, y)
                y_pred_train = gpr_pipeline.predict(X)
                r2_train = r2_score(y, y_pred_train)
                optimized_kernel = gpr_pipeline.named_steps['regressor'].kernel_
                
                results.append({
                    "Kernel": kernel_name,
                    "Init Length": length,
                    "Alpha": alpha,
                    "Train R2": r2_train,
                    "LOOCV R2": r2,
                    "LOOCV MAE": mae,
                    "LOOCV RMSE": rmse,
                    "Optimized Kernel": str(optimized_kernel)
                })
            except Exception as e:
                pass

df_res = pd.DataFrame(results).sort_values(by="LOOCV R2", ascending=False)
print("=== TOP 20 GPR MODELS ===")
print(df_res.head(20).to_string(index=False))
df_res.to_csv("c:\\Users\\CHUNG\\.gemini\\antigravity-ide\\scratch\\gpr_tuning_results.csv", index=False)
