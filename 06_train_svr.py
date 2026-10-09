# 06_train_svr.py
# Goal: train the FIRST BMI model = embeddings -> SVR (the paper's baseline method).
# Steps: load -> split -> scale -> train SVR -> predict on test.

import numpy as np 
from sklearn.svm import SVR                         # the Support Vector Regression model.
from sklearn.preprocessing import StandardScaler    # to put all features on a similar scale.

# --- 1) Load the saved embeddings
data = np.load("features/facenet_embeddings.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
is_training = data["is_training"]
gender = data["gender"]


# --- 2) Split into train / test using the is_training column ---
train_mask = is_training == 1
test_mask = is_training == 0
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Add gender as an extra feature column (1=male, 0=female).
# np.column_stack glues arrays side by side: (n, 512) + (n, 1) = (n, 513).
gender_train = gender[train_mask].reshape(-1, 1)        # shape: (3204, 1)
gender_test_arr = gender[test_mask].reshape(-1, 1)      # shape: (750, 1)
X_train = np.column_stack([X_train, gender_train])      # shape: (3204, 513)
X_test = np.column_stack([X_test, gender_test_arr])     # shape: (750, 513)
print("X_train with gender:", X_train.shape)


# --- 3) Scale the features ---
# fit_transform on TRAIN: learn the scaling rule from train, and apply it to train.
# transform on TEST: apply the SAME rule to test (do NOT learn from test).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 4) Train the SVR model ---
# C and gamma are settings that control the model; these are common starting values.
# We will tune them later. For now we just want a working baseline.
model = SVR(kernel="rbf", C=10.0, gamma="scale")
model.fit(X_train_scaled, y_train)  # This is the "studying" step.
print("training done.")


# --- 5) Predict BMI on the test set (faces the model has never seen) ---
y_pred = model.predict(X_test_scaled)

# Quick peek: show the first 5 predictions next to the true BMIs.
print("\nfirst 5 predictions vs true:")
for i in range(5):
    print(f"    predicted {y_pred[i]:.1f} | true {y_test[i]:.1f}")


# --- 6) Measure performance on the test set ---
from scipy.stats import pearsonr    # to compute the Pearson r correlation.
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Pearson r: how well predictions and true values move together (paper's metric).
# pearsonr returns two things; the first ([0]) is the correlation we want.
r = pearsonr(y_test, y_pred)[0]


# MAE: average absolute gap, in BMI points (easy to interpret).
mae = mean_absolute_error(y_test, y_pred)

# RMSE: like MAE but punishes big misses more. (sqrt of mean squared error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Test Performance ---")
print(f"Pearson r : {r:.3f} (paper baseline = 0.65)")
print(f"MAE       : {mae:.2f} BMI points")
print(f"RMSE      : {rmse:.2f} BMI points")


# --- 7) Break down performance by gender and BMI range ---
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# Load gender for the test set (1=male, 0=female).
gender = data["gender"]
gender_test = gender[test_mask]

# --- 7a) Performance by gender ---
print("\n--- by gender ---")
for label, value in [("Male", 1), ("Female", 0)]:
    # Build a mask for this gender within the test set.
    g_mask = gender_test == value

    # skip if too few samples.
    if g_mask.sum() < 10:
        print(f"{label}: too few samples")
        continue

    r_g = pearsonr(y_test[g_mask], y_pred[g_mask])[0]
    mae_g = mean_absolute_error(y_test[g_mask], y_pred[g_mask])
    print(f"   {label:7s}: r={r_g:.3f} MAE={mae_g:.2f} n={g_mask.sum()}")


    # --- 7b) Performance by BMI range ---
print("\n--- by BMI range ---")
bmi_ranges = [
        ("Underweight", 0, 18.5),
        ("Normal", 18.5, 25),
        ("Overweight", 25, 30),
        ("Obese", 30, 999),
]

for range_label, low, high in bmi_ranges:
    # Mask: test samples whose TRUE bmi falls in this range.
    b_mask = (y_test >= low) & (y_test < high)
    count = b_mask.sum()

    if count < 10:
         print(f" {range_label:12s}: too few samples (n={count})")
         continue

    mae_b = mean_absolute_error(y_test[b_mask], y_pred[b_mask])
    print(f"  {range_label:12s}: MAE={mae_b:.2f}  n={count}")




