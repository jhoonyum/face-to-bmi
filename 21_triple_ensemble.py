# 21_triple_ensemble.py
# FaceNet frozen(512) + ArcFace frozen(512) + FaceNet finetuned(512) = 1536차원 SVR.
# 20_extract_finetuned_embeddings.py를 먼저 실행해서 finetuned_v2_embeddings.npz를 만들어야 함.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1) 세 임베딩 불러오기 ----
fn = np.load("features/facenet_embeddings.npz",       allow_pickle=True)
ac = np.load("features/arcface_embeddings.npz",       allow_pickle=True)
ft = np.load("features/finetuned_v2_embeddings.npz",  allow_pickle=True)

# ---- 2) 세 파일에 공통으로 있는 얼굴만 (inner join) ----
ac_name_set = {name: i for i, name in enumerate(ac["names"])}
ft_name_set = {name: i for i, name in enumerate(ft["names"])}

fn_idx, ac_idx, ft_idx = [], [], []
for i, name in enumerate(fn["names"]):
    if name in ac_name_set and name in ft_name_set:
        fn_idx.append(i)
        ac_idx.append(ac_name_set[name])
        ft_idx.append(ft_name_set[name])

fn_idx = np.array(fn_idx)
ac_idx = np.array(ac_idx)
ft_idx = np.array(ft_idx)

print(f"FaceNet: {len(fn['names'])}개, ArcFace: {len(ac['names'])}개, "
      f"Finetuned: {len(ft['names'])}개, 공통: {len(fn_idx)}개")

# ---- 3) 이어붙이기: 512 + 512 + 512 = 1536차원 ----
X_fn = fn["X"][fn_idx]
X_ac = ac["X"][ac_idx]
X_ft = ft["X"][ft_idx]
y    = fn["y"][fn_idx]
is_training = fn["is_training"][fn_idx]

X = np.concatenate([X_fn, X_ac, X_ft], axis=1)
print(f"triple ensemble shape: {X.shape}")

# ---- 4) train / test 분리 ----
train_mask = is_training == 1
test_mask  = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"train: {X_train.shape[0]}개, test: {X_test.shape[0]}개")

# ---- 5) C값 튜닝 ----
C_values = [1.0, 5.0, 10.0, 50.0, 100.0]
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

print(f"\nbest C = {best_C}")

# ---- 6) 최종 평가 ----
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf", C=best_C, gamma="scale")),
])
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

r    = pearsonr(y_test, preds)[0]
mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n--- Triple Ensemble Results ---")
print(f"Pearson r : {r:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print("\n--- Comparison ---")
print("ArcFace frozen (C=1.0)        : r = 0.690")
print("ArcFace frozen (C=10.0)       : r = 0.714")
print("Ensemble x2   (C=10.0)        : r = 0.721")
print(f"Triple ensemble (C={best_C}) : r = {r:.3f}")
