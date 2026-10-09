# 16_save_arcface_model.py
# Goal: train the best model (ArcFace embeddings + SVR) and SAVE it to a file,
# so the Streamlit demo can load it instantly without retraining every time.
#
# This is a one-time script. Run it once; it creates arcface_svr_model.pkl.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib   # joblib saves/loads scikit-learn models to a file (like a "freezer")

# --- 1) Load the pre-extracted ArcFace embeddings ---
data = np.load("features/arcface_embeddings.npz", allow_pickle=True)
X = data["X"]              # ArcFace embeddings, shape (num_faces, 512)
y = data["y"]              # true BMI values
is_training = data["is_training"]

# --- 2) Use ALL training data to build the final model ---
# (For the demo we want the best possible model, so we train on the train split.)
train_mask = is_training == 1
X_train = X[train_mask]
y_train = y[train_mask]
print("training on", X_train.shape[0], "faces")

# --- 3) Build a Pipeline: scaler + SVR in one object ---
# A Pipeline bundles the scaler and the model together, so when we load it later,
# scaling happens automatically before prediction. This avoids mistakes.
# C=1.0 was the best value we found during cross-validation.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=1.0, gamma="scale")),
])

# --- 4) Train the model ---
model.fit(X_train, y_train)
print("training done.")

# --- 5) Save the trained model to a file ---
# joblib.dump writes the whole pipeline (scaler + svr) into one .pkl file.
joblib.dump(model, "arcface_svr_model.pkl")
print("saved model to arcface_svr_model.pkl")
