# 23_ensemble_arcface_adaface.py
# ArcFace 512 + AdaFace 512 = 1024차원 앙상블. C값 튜닝 포함.
# AdaFace가 ArcFace에 새 정보를 더하는지 확인.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1) 두 임베딩 불러오기 ----
ac = np.load("features/arcface_embeddings.npz", allow_pickle=True)
ad = np.load("features/adaface_embeddings.npz", allow_pickle=True)

# ---- 2) 공통 얼굴만 (inner join on name) ----
ad_name_set = {name: i for i, name in enumerate(ad["names"])}
ac_idx, ad_idx = [], []
for i, name in enumerate(ac["names"]):
    if name in ad_name_set:
        ac_idx.append(i)
        ad_idx.append(ad_name_set[name])

ac_idx = np.array(ac_idx)
ad_idx = np.array(ad_idx)
print(f"ArcFace: {len(ac['names'])}, AdaFace: {len(ad['names'])}, 공통: {len(ac_idx)}")

X_ac = ac["X"][ac_idx]
X_ad = ad["X"][ad_idx]
y    = ac["y"][ac_idx]
is_training = ac["is_training"][ac_idx]

# ---- 3) 이어붙이기 ----
X = np.concatenate([X_ac, X_ad], axis=1)
print(f"ensemble shape: {X.shape}")

train_mask = is_training == 1
test_mask  = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")

# ---- 4) C값 튜닝 ----
C_values = [1.0, 5.0, 10.0, 50.0, 100.0]
print("\nSearching best C...")
print(f"{'C':>8}  {'CV MAE':>8}")
best_C, best_score = None, float("inf")
for C in C_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=C, gamma="scale")),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5,
                             scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    print(f"{C:>8.2f}  {mae:>8.4f}")
    if mae < best_score:
        best_score, best_C = mae, C
print(f"\nbest C = {best_C}")

# ---- 5) 최종 평가 ----
final = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf", C=best_C, gamma="scale")),
])
final.fit(X_train, y_train)
preds = final.predict(X_test)

r    = pearsonr(y_test, preds)[0]
mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n--- ArcFace + AdaFace Ensemble ---")
print(f"Pearson r : {r:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print("\n--- Comparison ---")
print("ArcFace alone (C=10)            : r = 0.714")
print("FaceNet + ArcFace (C=10)        : r = 0.721")
print(f"ArcFace + AdaFace (C={best_C})  : r = {r:.3f}")
