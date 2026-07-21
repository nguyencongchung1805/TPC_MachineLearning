import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
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

# Define models to test
# 1. Baseline RSM
evaluate_pipeline("RSM (LinearRegression + Poly2)", Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
]))

# 2. Regularized RSM models (Ridge, Lasso, ElasticNet)
for alpha in [0.01, 0.1, 1.0, 10.0]:
    evaluate_pipeline(f"Ridge (alpha={alpha}) + Poly2", Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', Ridge(alpha=alpha))
    ]))
    
    evaluate_pipeline(f"Lasso (alpha={alpha}) + Poly2", Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', Lasso(alpha=alpha, max_iter=10000))
    ]))

    evaluate_pipeline(f"ElasticNet (alpha={alpha}) + Poly2", Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000))
    ]))

# 3. SVR
for kernel in ['linear', 'poly', 'rbf']:
    for C_val in [0.1, 1.0, 10.0, 100.0]:
        for epsilon in [0.01, 0.1, 0.2]:
            evaluate_pipeline(f"SVR ({kernel}, C={C_val}, eps={epsilon})", Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', SVR(kernel=kernel, C=C_val, epsilon=epsilon))
            ]))

# 4. SVR with Poly Features
for kernel in ['linear', 'rbf']:
    for C_val in [0.1, 1.0, 10.0, 100.0]:
        evaluate_pipeline(f"SVR + Poly2 ({kernel}, C={C_val})", Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', SVR(kernel=kernel, C=C_val))
        ]))

# 5. Gaussian Process Regression
kernels = [
    C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
    C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5),
    C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5),
    RBF(1.0),
    Matern(nu=1.5)
]
for i, k in enumerate(kernels):
    evaluate_pipeline(f"GPR (kernel_{i})", Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GaussianProcessRegressor(kernel=k, alpha=0.1, n_restarts_optimizer=10, random_state=42))
    ]))

# 6. PLS Regression
for n_comp in [1, 2, 3, 4]:
    evaluate_pipeline(f"PLS (n_components={n_comp})", Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', PLSRegression(n_components=n_comp))
    ]))

# 7. Ensemble Models (with Tuning)
evaluate_pipeline("Random Forest (default)", Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
]))
evaluate_pipeline("Random Forest (tuned: max_depth=3, n_estimators=50)", Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(max_depth=3, n_estimators=50, random_state=42))
]))

evaluate_pipeline("Extra Trees (default)", Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', ExtraTreesRegressor(random_state=42))
]))

evaluate_pipeline("Gradient Boosting (default)", Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
]))
evaluate_pipeline("Gradient Boosting (tuned: max_depth=2, n_estimators=50)", Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(max_depth=2, n_estimators=50, learning_rate=0.05, random_state=42))
]))

evaluate_pipeline("AdaBoost (default)", Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', AdaBoostRegressor(random_state=42))
]))

# 8. XGBoost (tuned)
for lr in [0.01, 0.05, 0.1]:
    for max_depth in [2, 3]:
        for n_estimators in [20, 50, 100]:
            evaluate_pipeline(f"XGBoost (lr={lr}, depth={max_depth}, n_est={n_estimators})", Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', XGBRegressor(learning_rate=lr, max_depth=max_depth, n_estimators=n_estimators, random_state=42, objective='reg:squarederror'))
            ]))

# Sort results by LOOCV R2 descending
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="LOOCV R2", ascending=False)
print("=== RESULTS SUMMARY (Top 30 models by LOOCV R2) ===")
print(df_results.head(30).to_string(index=False))
df_results.to_csv("c:\\Users\\CHUNG\\.gemini\\antigravity-ide\\scratch\\evaluation_results.csv", index=False)
