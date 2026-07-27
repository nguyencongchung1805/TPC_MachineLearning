import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

xlsx_path = r"d:\do an\TPC_MachineLearning\data\tpc_dataset.xlsx"
df = pd.read_excel(xlsx_path)

# Extract features and target
X = df.drop(columns=["Run", "TPC (mg GAE/g)"])
y = df["TPC (mg GAE/g)"]

loo = LeaveOneOut()

results = []

def evaluate_pipeline(name, model_pipeline):
    try:
        y_pred = cross_val_predict(model_pipeline, X, y, cv=loo)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Also fit on whole data to get training R2
        model_pipeline.fit(X, y)
        y_pred_train = model_pipeline.predict(X)
        r2_train = r2_score(y, y_pred_train)
        
        results.append({
            "Model": name,
            "Train R2": r2_train,
            "LOOCV R2": r2,
            "LOOCV MAE": mae,
            "LOOCV RMSE": rmse
        })
    except Exception as e:
        print(f"Failed to run {name}: {str(e)}")

# 1. Baseline RSM (2nd-Order Polynomial)
evaluate_pipeline("RSM (LinearRegression + Poly2)", Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
]))

# 2. Gaussian Process Regression (GPR) Configurations
gpr_kernels = [
    ("GPR (RBF, alpha=0.05)", C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), 0.05),
    ("GPR (RBF, alpha=0.01)", C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), 0.01),
    ("GPR (RBF, alpha=0.1)", C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), 0.1),
    ("GPR (Matern nu=2.5, alpha=0.05)", C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5), 0.05),
    ("GPR (Matern nu=1.5, alpha=0.05)", C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5), 0.05)
]

for name, k, a in gpr_kernels:
    evaluate_pipeline(name, Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GaussianProcessRegressor(kernel=k, alpha=a, n_restarts_optimizer=10, random_state=42))
    ]))

# Summary
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="LOOCV R2", ascending=False)
print("=== RESULTS SUMMARY (RSM vs GPR Models) ===")
print(df_results.to_string(index=False))
df_results.to_csv(r"d:\do an\TPC_MachineLearning\results\evaluation_results.csv", index=False)

