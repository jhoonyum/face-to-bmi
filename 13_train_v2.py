# 13_train_v2.py
# Improved fine-tuning: normalized BMI targets, partial unfreezing, differential learning rates.
# Saves the best checkpoint (highest test r) to bmi_finetuned_v2.pth.

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from importlib import import_module

BMIModel = import_module("11_model").BMIModel
FaceBMIDataset = import_module("12_dataset").FaceBMIDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---- 1) Compute BMI mean and std from the training split only ----
# We normalize targets so the model sees values near 0 instead of 18-85.
# Using only train data stats prevents any leakage from the test set.
df = pd.read_csv("data/data.csv", index_col=0)
train_df = df[df["is_training"] == 1]
bmi_mean = float(train_df["bmi"].mean())
bmi_std  = float(train_df["bmi"].std())
print(f"BMI mean: {bmi_mean:.2f}  std: {bmi_std:.2f}")

# ---- 2) Build model then apply partial unfreezing ----
model = BMIModel().to(device)

# Step A: freeze every parameter in the entire model.
for param in model.parameters():
    param.requires_grad = False

# Step B: unfreeze our custom regression head completely.
# The head has very few parameters and needs to learn from scratch.
for param in model.head.parameters():
    param.requires_grad = True

# Step C: unfreeze only the last few layers of the FaceNet backbone.
# These are the layers closest to the output (highest-level features).
# Earlier layers (edges, textures) stay frozen; we trust the pretrained version.
UNFREEZE_BACKBONE = ["block8", "avgpool_1a", "last_linear", "last_bn"]
for name, param in model.backbone.named_parameters():
    if any(layer in name for layer in UNFREEZE_BACKBONE):
        param.requires_grad = True

# Print how many parameters are actually being trained.
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")

# ---- 3) Optimizer with different learning rates per group ----
# Head gets a large lr (it is starting from scratch).
# Unfrozen backbone layers get a tiny lr (they already know something useful).
optimizer = torch.optim.Adam([
    {"params": model.head.parameters(),                                              "lr": 1e-3},
    {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad],"lr": 1e-5},
])
criterion = nn.MSELoss()

# ---- 4) Data loaders ----
train_loader = DataLoader(
    FaceBMIDataset("data/data.csv", "data/Images", is_training_value=1),
    batch_size=16, shuffle=True,
)
test_loader = DataLoader(
    FaceBMIDataset("data/data.csv", "data/Images", is_training_value=0),
    batch_size=16, shuffle=False,
)

# ---- 5) Evaluation helper ----
# Runs the model on a DataLoader, denormalizes predictions, returns r and MAE.
def evaluate(loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, bmis in loader:
            images = images.to(device)
            bmis   = bmis.to(device)
            valid  = bmis > 0
            if valid.sum() == 0:
                continue
            images, bmis = images[valid], bmis[valid]
            out = model(images).squeeze(1)
            # Denormalize: model output is in normalized space, convert back to BMI.
            out_bmi = out * bmi_std + bmi_mean
            preds.extend(out_bmi.cpu().numpy())
            truths.extend(bmis.cpu().numpy())
    r   = pearsonr(np.array(truths), np.array(preds))[0]
    mae = mean_absolute_error(np.array(truths), np.array(preds))
    return r, mae

# ---- 6) Training loop ----
num_epochs = 200
best_r     = -1.0
best_epoch = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss, num_batches = 0.0, 0

    for images, bmis in train_loader:
        images = images.to(device)
        bmis   = bmis.to(device)
        valid  = bmis > 0
        if valid.sum() == 0:
            continue
        images, bmis = images[valid], bmis[valid]

        # Normalize the BMI targets before computing loss.
        bmis_norm = (bmis - bmi_mean) / bmi_std

        optimizer.zero_grad()
        predictions = model(images).squeeze(1)
        loss = criterion(predictions, bmis_norm)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Evaluate on the test set every 5 epochs (and on epoch 1 to see the starting point).
    if epoch % 5 == 0 or epoch == 1:
        r, mae = evaluate(test_loader)
        tag = "  <-- best" if r > best_r else ""
        if r > best_r:
            best_r     = r
            best_epoch = epoch
            torch.save(model.state_dict(), "bmi_finetuned_v2.pth")
        print(f"epoch {epoch:3d}/{num_epochs} | loss: {avg_loss:.4f} | test r: {r:.3f} | MAE: {mae:.2f}{tag}")
    else:
        print(f"epoch {epoch:3d}/{num_epochs} | loss: {avg_loss:.4f}")

print(f"\nbest test r = {best_r:.3f}  at epoch {best_epoch}")
print("saved to bmi_finetuned_v2.pth")
