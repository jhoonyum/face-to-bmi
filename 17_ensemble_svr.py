# 17_ensemble_svr.py
# FaceNet 512차원 + ArcFace 512차원을 이어붙여 1024차원으로 SVR 학습.
# 두 임베딩이 서로 다른 얼굴 정보를 담고 있다면 합쳤을 때 더 올라간다.
# 두 npz 파일에 공통으로 있는 얼굴만 사용 (inner join on name).

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1) 두 임베딩 파일 불러오기 ----
fn = np.load("features/facenet_embeddings.npz", allow_pickle=True)
ac = np.load("features/arcface_embeddings.npz", allow_pickle=True)

fn_names = fn["names"]
ac_names = ac["names"]

# ---- 2) 공통 얼굴만 고르기 (inner join) ----
# FaceNet은 3,954개, ArcFace는 3,208개를 검출했다.
# 둘 다 검출된 얼굴만 사용해야 임베딩을 이어붙일 수 있다.
ac_name_set = {name: i for i, name in enumerate(ac_names)}
common_fn_idx, common_ac_idx = [], []

for i, name in enumerate(fn_names):
    if name in ac_name_set:
        common_fn_idx.append(i)
        common_ac_idx.append(ac_name_set[name])

common_fn_idx = np.array(common_fn_idx)
common_ac_idx = np.array(common_ac_idx)

print(f"FaceNet: {len(fn_names)}개, ArcFace: {len(ac_names)}개, 공통: {len(common_fn_idx)}개")

# 공통 얼굴의 임베딩과 라벨 추출
X_fn = fn["X"][common_fn_idx]          # (n, 512)
X_ac = ac["X"][common_ac_idx]          # (n, 512)
y    = fn["y"][common_fn_idx]          # BMI 값
is_training = fn["is_training"][common_fn_idx]

# ---- 3) 이어붙이기: 512 + 512 = 1024차원 ----
X = np.concatenate([X_fn, X_ac], axis=1)
print(f"앙상블 임베딩 shape: {X.shape}")

# ---- 4) train / test 분리 ----
train_mask = is_training == 1
test_mask  = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"train: {X_train.shape[0]}개, test: {X_test.shape[0]}개")

# ---- 5) SVR 학습 및 평가 ----
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf", C=1.0, gamma="scale")),
])
model.fit(X_train, y_train)
preds = model.predict(X_test)

r    = pearsonr(y_test, preds)[0]
mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n--- Ensemble (FaceNet + ArcFace) Results ---")
print(f"Pearson r : {r:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print("\n--- Comparison ---")
print("FaceNet frozen + SVR  : r = 0.577")
print("ArcFace frozen + SVR  : r = 0.690")
print(f"Ensemble (1024-dim)   : r = {r:.3f}")
