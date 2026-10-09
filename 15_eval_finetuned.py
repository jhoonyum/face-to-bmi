# 15_eval_finetuned.py
# Goal: evaluate the fine-tuned model on the TEST set (faces it never saw),
# and compare its r/MAE against our frozen models.

import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Bring in the model class and the dataset class we already built.
from importlib import import_module
BMIModel = import_module("11_model").BMIModel
FaceBMIDataset = import_module("12_dataset").FaceBMIDataset

# --- 1) Set up device and load the trained model ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# Build an empty model with the same structure, then pour in the saved weights.
model = BMIModel().to(device)
model.load_state_dict(torch.load("bmi_finetuned.pth", map_location=device))
model.eval()   # use mode (not training)
print("loaded trained model.")

# --- 2) Build the TEST dataset and loader ---
# is_training_value=0 -> the test split (faces the model never trained on).
# shuffle=False -> no need to shuffle when only evaluating.
test_dataset = FaceBMIDataset(
    csv_path="data/data.csv",
    images_dir="data/Images",
    is_training_value=0,
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- 3) Run the model over all test batches and collect predictions ---
all_predictions = []   # will hold every predicted BMI
all_truths = []        # will hold every true BMI (in the same order)

# torch.no_grad: we are only using the model, not training -> faster, less memory.
with torch.no_grad():
    for images, bmis in test_loader:
        images = images.to(device)
        bmis = bmis.to(device)

        # Filter out invalid samples (no face / missing file -> bmi was set to -1).
        valid = bmis > 0
        if valid.sum() == 0:
            continue
        images = images[valid]
        bmis = bmis[valid]

        # Predict, then reshape (batch,1) -> (batch,) to match the truths.
        predictions = model(images).squeeze(1)

        # Move results back to cpu and store them as plain numbers.
        all_predictions.extend(predictions.cpu().numpy())
        all_truths.extend(bmis.cpu().numpy())

# Turn the collected lists into numpy arrays for the metric functions.
all_predictions = np.array(all_predictions)
all_truths = np.array(all_truths)
print("evaluated on", len(all_truths), "test faces")


# --- DIAGNOSTIC: print first 10 predictions vs truths to check pairing ---
print("\n--- diagnostic: pred vs truth (first 10) ---")
for i in range(10):
    print(f"  pred {all_predictions[i]:.1f}  |  truth {all_truths[i]:.1f}")


# --- 4) Compute the metrics ---
r    = pearsonr(all_truths, all_predictions)[0]
mae  = mean_absolute_error(all_truths, all_predictions)
rmse = np.sqrt(mean_squared_error(all_truths, all_predictions))

print("\n--- Fine-tuned Model Test Performance ---")
print(f"Pearson r : {r:.3f}")
print(f"MAE       : {mae:.2f}  BMI points")
print(f"RMSE      : {rmse:.2f}  BMI points")

print("\n--- Comparison ---")
print("paper baseline      : r = 0.650")
print("FaceNet frozen + SVR: r = 0.577")
print("ArcFace frozen + SVR: r = 0.690")
print(f"FaceNet fine-tuned  : r = {r:.3f}")
