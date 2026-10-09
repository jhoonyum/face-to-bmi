# 27_final_ensemble.py
# Compare ALL combinations (1, 2, 3, 4 models) of the four embeddings:
# FaceNet, ArcFace, TopoFR, LVFace.
# Total 15 combos: 4 singles + 6 pairs + 4 triples + 1 all-four.
# Each combo: tune C, then measure test Pearson r. Uses only faces common to all four.

import numpy as np
from itertools import combinations
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# ---- 1) load the four embeddings ----
sources = {
    "FaceNet": np.load("features/facenet_embeddings.npz", allow_pickle=True),
    "ArcFace": np.load("features/arcface_embeddings.npz", allow_pickle=True),
    "TopoFR":  np.load("features/topofr_embeddings.npz",  allow_pickle=True),
    "LVFace":  np.load("features/lvface_embeddings.npz",  allow_pickle=True),
}

# ---- 2) keep only faces present in ALL four files (4-way inner join) ----
name_to_idx = {k: {n: i for i, n in enumerate(v["names"])} for k, v in sources.items()}

common_names = []
for n in sources["FaceNet"]["names"]:
    if all(n in name_to_idx[k] for k in sources):
        common_names.append(n)
print("common faces:", len(common_names))

def get_matrix(key):
    idx = np.array([name_to_idx[key][n] for n in common_names])
    return sources[key]["X"][idx]

embeds = {k: get_matrix(k) for k in sources}

fn_idx = np.array([name_to_idx["FaceNet"][n] for n in common_names])
y = sources["FaceNet"]["y"][fn_idx]
is_training = sources["FaceNet"]["is_training"][fn_idx]
train_mask = is_training == 1
test_mask = is_training == 0

# ---- 3) evaluate one combination ----
def evaluate_combo(keys):
    X = np.concatenate([embeds[k] for k in keys], axis=1)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    best_C, best_cv = None, float("inf")
    for C in [5.0, 10.0, 50.0]:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(kernel="rbf", C=C, gamma="scale"))])
        cv = -cross_val_score(pipe, X_train, y_train, cv=5,
                              scoring="neg_mean_absolute_error").mean()
        if cv < best_cv:
            best_cv, best_C = cv, C

    final = Pipeline([("scaler", StandardScaler()),
                      ("svr", SVR(kernel="rbf", C=best_C, gamma="scale"))])
    final.fit(X_train, y_train)
    preds = final.predict(X_test)
    r = pearsonr(y_test, preds)[0]
    mae = mean_absolute_error(y_test, preds)
    return r, mae, best_C

# ---- 4) loop over every combination size 1..4 ----
keys = ["FaceNet", "ArcFace", "TopoFR", "LVFace"]
results = []

print("\n{:<42} {:>7} {:>7} {:>6}".format("Combination", "r", "MAE", "C"))
print("=" * 64)

for size in range(1, len(keys) + 1):
    for combo in combinations(keys, size):
        r, mae, C = evaluate_combo(list(combo))
        label = " + ".join(combo)
        results.append((label, r, mae, C))
        print("{:<42} {:>7.3f} {:>7.2f} {:>6.0f}".format(label, r, mae, C))
    print("-" * 64)

# ---- 5) ranked by Pearson r ----
print("\n=== Ranked by Pearson r (best first) ===")
print("{:<5} {:<42} {:>7} {:>7}".format("Rank", "Combination", "r", "MAE"))
print("-" * 64)
for rank, (label, r, mae, C) in enumerate(
        sorted(results, key=lambda x: x[1], reverse=True), start=1):
    print("{:<5} {:<42} {:>7.3f} {:>7.2f}".format(rank, label, r, mae))
