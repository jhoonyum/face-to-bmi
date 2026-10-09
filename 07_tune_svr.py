# 07_tune_svr.py
# Goal: find the best C value for SVR using cross-validation on the TRAIN set.
# We never touch the test set during tuning

import numpy as np 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


# --- 1) Load and split (same as before, without gender it had no effect) ---
data = np.load("features/facenet_embeddings.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
is_training = data["is_training"]

train_mask = is_training == 1
test_mask = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]


# --- 2) Try several C values, score each with 3-fold CV on the train set ---
# A pipeline chains scaler + model into one object.
# This way, each CV fold scales using only that fold's train data (correct).
C_candidates = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]

print("Searching best C (3-fold CV on train)...")
print(f"{'C':>8}  {'CV MAE':>8}  {'std':>8}")
print("-" * 30)

best_c = None
best_score = -999

for c in C_candidates:
    # Pipeline: always scale first, then apply SVR. Done in one object.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=c, gamma="scale")),
    ])

    # cross_val_score runs K-fold CV and return K scores.
    # scoring="r2" is not Pearson r, but it is correlated and fast.
    # We use negative MAE as proxy (neg_mean_absolute_error) for speed.
    # cv=3 means 3-fold (faster than 5 on big data).

    scores = cross_val_score(
            pipe, X_train, y_train,
            cv=3, scoring="neg_mean_absolute_error"
    )

    # neg_mean_absolute_error is negative (sklearn convention)
    # therefore, FLIP SIGN
    mean_mae = -scores.mean()
    std_mae = scores.std()
    print(f"{c:>8}  {mean_mae:>8.3f}  {std_mae:>8.3f}")

    if mean_mae < best_score * -1 or best_c is None:
        best_score = -mean_mae
        best_c = c

print(f"\nbest c found: {best_c}")


# --- 3) Retrain on ALL train data with the best C. then evaluate on test ---
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=best_c, gamma="scale")),
])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

r = pearsonr(y_test, y_pred)[0]
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Test Performance (best_c) ---")
print(f"Pearson r: {r:.3f} (paper baseline = 0.65)")
print(f"MAE      : {mae:.2f} BMI points")
