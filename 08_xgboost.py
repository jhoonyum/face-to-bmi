# 08_xgboost.py
# Goal: try XGBoost instead of SVR, same embeddings.
# Question we are answering: is the regressor the bottleneck, or the embedding?

import numpy as np 
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load and split ---
data = np.load("features/facenet_embeddings.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
is_training = data["is_training"]

train_mask = is_training == 1
test_mask = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print("train:", X_train.shape, "| test:", X_test.shape)


# --- 2) Train XGBoost ---
# Note: tree models like XGBoost do NOT need feature scaling.
# They split on thresholds, so the scale of each feature does not matter.
# So no StandardScaler here.
# n_estimators = how many trees to stack.
# max_depth = how complex each tree is.
# These are common starting values: we can tune later if it looks promising.
model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        )
model.fit(X_train, y_train)
print("training done.")

# --- 3) Evaluate on test ---
y_pred = model.predict(X_test)

r    = pearsonr(y_test, y_pred)[0]
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Test Performance (XGBRegressor) ---")
print(f"Pearson r : {r:.3f}     (paper baseline = 0.65, SVR was 0.577)")
print(f"MAE       : {mae:.3f}   BMI points")
print(f"RMSE      : {rmse:.3f}  BMI points")
