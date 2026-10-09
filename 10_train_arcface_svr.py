# 10_train_arcface_svr.py
# Goal: train SVR on ARCFACE embeddings, compare against FaceNet's r=0.577.
# IMPORTANT: ArcFace embeddings mean different things than FaceNet's 
# Therefore, we must train a FRESH regressor on them.
# We cannot reuse the FaceNet-trained model.

import numpy as np 
from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --- 1) Load the ARCFACE embeddings ---
data = np.load("features/arcface_embeddings.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
is_training = data["is_training"]

# --- 2) Split using the same is_training column ---
train_mask = is_training == 1
test_mask = is_training == 0 
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print("ArcFace -> train:", X_train.shape, "| test:", X_test.shape)

# --- 3) Scale ---
# ArcFace embeddings also benefit from scaling for SVR ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4) Train SVR ---
# Use C=1.0 (The best C we found for FaceNet) as a fair start. ---
model = SVR(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train_scaled, y_train)
print("training done.")

# --- 5) Evaluate on test ---
y_pred = model.predict(X_test_scaled)
r      = pearsonr(y_test, y_pred)[0]
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n---ArcFace Test Performance ---")
print(f"Pearson r : {r:.3f}     (paper 0.65 | FaceNet was 0.577)")
print(f"MAE       : {mae:.2f} BMI points")
print(f"RMSE      : {rmse:.2f} BMI points")

