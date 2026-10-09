# 25_triple_diverse_ensemble.py
# 서로 다른 세 계열: FaceNet(triplet) + ArcFace(각도마진) + TopoFR(위상정렬)
# 512 x 3 = 1536차원. C값 튜닝 포함.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1) 세 임베딩 불러오기 ----
fn = np.load("features/facenet_embeddings.npz", allow_pickle=True)
ac = np.load("features/arcface_embeddings.npz", allow_pickle=True)
tp = np.load("features/topofr_embeddings.npz",  allow_pickle=True)

# ---- 2) 세 파일 공통 얼굴만 (inner join) ----
ac_set = {n: i for i, n in enumerate(ac["names"])}
tp_set = {n: i for i, n in enumerate(tp["names"])}

fn_idx, ac_idx, tp_idx = [], [], []
for i, n in enumerate(fn["names"]):
    if n in ac_set and n in tp_set:
        fn_idx.append(i)
        ac_idx.append(ac_set[n])
        tp_idx.append(tp_set[n])

fn_idx, ac_idx, tp_idx = map(np.array, (fn_idx, ac_idx, tp_idx))
print(f"FaceNet {len(fn['names'])}, ArcFace {len(ac['names'])}, "
      f"TopoFR {len(tp['names'])}, 공통: {len(fn_idx)}")

X_fn = fn["X"][fn_idx]
X_ac = ac["X"][ac_idx]
X_tp = tp["X"][tp_idx]
y    = fn["y"][fn_idx]
is_training = fn["is_training"][fn_idx]

# ---- 3) 이어붙이기: 1536차원 ----
X = np.concatenate([X_fn, X_ac, X_tp], axis=1)
print(f"triple diverse shape: {X.shape}")

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

print("\n--- Triple Diverse Ensemble (FaceNet + ArcFace + TopoFR) ---")
print(f"Pearson r : {r:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print("\n--- Comparison ---")
print("ArcFace alone (C=10)        : r = 0.714")
print("ArcFace + AdaFace (C=10)    : r = 0.724")
print(f"FaceNet+ArcFace+TopoFR (C={best_C}) : r = {r:.3f}")
