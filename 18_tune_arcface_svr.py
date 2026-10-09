# 18_tune_arcface_svr.py
# ArcFace 임베딩에 대해 SVR C값을 cross-validation으로 최적화.
# ArcFace 단독으로 C 튜닝했을 때 얼마나 나오는지 확인.
# 앙상블(r=0.721)이 진짜 앙상블 효과인지, C 튜닝 효과인지 구분하기 위해.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1) ArcFace 임베딩 불러오기 ----
data = np.load("features/arcface_embeddings.npz", allow_pickle=True)
X, y, is_training = data["X"], data["y"], data["is_training"]

train_mask = is_training == 1
test_mask  = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"train: {X_train.shape[0]}개, test: {X_test.shape[0]}개, dims: {X.shape[1]}")

# ---- 2) C값 탐색 ----
C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
print("\nSearching best C value...")
print(f"{'C':>8}  {'CV MAE':>8}")

best_C, best_score = None, float("inf")
for C in C_values:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=C, gamma="scale")),
    ])
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring="neg_mean_absolute_error")
    mean_mae = -scores.mean()
    print(f"{C:>8.2f}  {mean_mae:>8.4f}")
    if mean_mae < best_score:
        best_score = mean_mae
        best_C = C

print(f"\nbest C = {best_C}  (CV MAE = {best_score:.4f})")

# ---- 3) 최적 C로 최종 평가 ----
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf", C=best_C, gamma="scale")),
])
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

r    = pearsonr(y_test, preds)[0]
mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n--- ArcFace + SVR (tuned C) Results ---")
print(f"Pearson r : {r:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print("\n--- Comparison ---")
print("ArcFace + SVR (C=1.0, default)  : r = 0.690")
print(f"ArcFace + SVR (C={best_C}, tuned) : r = {r:.3f}")
print("Ensemble   (C=10.0, tuned)      : r = 0.721")
