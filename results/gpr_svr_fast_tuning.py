import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

xlsx_path = r"d:\do an\TPC_MachineLearning\data\tpc_dataset.xlsx"
df = pd.read_excel(xlsx_path)

X = df.drop(columns=["Run", "TPC (mg GAE/g)"])
y = df["TPC (mg GAE/g)"]

loo = LeaveOneOut()

results = []

# Focused GPR search
alphas = [0.01, 0.05, 0.1, 0.15, 0.2]
lengths = [2.0, 5.0, 10.0, 15.0, 20.0]
for nu in [1.5, 2.5]:
    for length in lengths:
        for alpha in alphas:
            k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length, length_scale_bounds=(1e-2, 1e2), nu=nu)
            gpr = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GaussianProcessRegressor(kernel=k, alpha=alpha, n_restarts_optimizer=5, random_state=42))
            ])
            y_pred = cross_val_predict(gpr, X, y, cv=loo)
            r2 = r2_score(y, y_pred)
            results.append({
                "Model": f"GPR (Matern {nu}, len={length}, alpha={alpha})",
                "LOOCV R2": r2,
                "LOOCV MAE": mean_absolute_error(y, y_pred),
                "LOOCV RMSE": np.sqrt(mean_squared_error(y, y_pred))
            })

for length in lengths:
    for alpha in alphas:
        k = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length, length_scale_bounds=(1e-2, 1e2))
        gpr = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GaussianProcessRegressor(kernel=k, alpha=alpha, n_restarts_optimizer=5, random_state=42))
        ])
        y_pred = cross_val_predict(gpr, X, y, cv=loo)
        r2 = r2_score(y, y_pred)
        results.append({
            "Model": f"GPR (RBF, len={length}, alpha={alpha})",
            "LOOCV R2": r2,
            "LOOCV MAE": mean_absolute_error(y, y_pred),
            "LOOCV RMSE": np.sqrt(mean_squared_error(y, y_pred))
        })

# Focused SVR search
for C_val in [1.0, 5.0, 10.0, 20.0, 50.0]:
    for epsilon in [0.05, 0.1, 0.2]:
        for gamma in ['scale', 'auto', 0.1, 0.05]:
            svr = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', SVR(kernel='rbf', C=C_val, epsilon=epsilon, gamma=gamma))
            ])
            y_pred = cross_val_predict(svr, X, y, cv=loo)
            r2 = r2_score(y, y_pred)
            results.append({
                "Model": f"SVR (RBF, C={C_val}, eps={epsilon}, g={gamma})",
                "LOOCV R2": r2,
                "LOOCV MAE": mean_absolute_error(y, y_pred),
                "LOOCV RMSE": np.sqrt(mean_squared_error(y, y_pred))
            })

df_res = pd.DataFrame(results).sort_values(by="LOOCV R2", ascending=False)
print("=== TOP 15 MODELS FROM FOCUSED TUNING ===")
print(df_res.head(15).to_string(index=False))
df_res.to_csv("c:\\Users\\CHUNG\\.gemini\\antigravity-ide\\scratch\\focused_tuning_results.csv", index=False)
