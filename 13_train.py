# 13_train.py
# Goal: fine-tune the model (backbone + head) on the BMI task.
# The training loop: for each batch -> predict -> score -> find fixes -> apply fixes.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# Bring in the things we already built in earlier files.
from importlib import import_module
BMIModel = import_module("11_model").BMIModel
FaceBMIDataset = import_module("12_dataset").FaceBMIDataset


# --- 1) Setup: device, model, data ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# Build the model (backbone + head) and move it onto the device.
model = BMIModel().to(device)

# Build the training dataset and its loader (batches of 16, shuffled).
train_dataset = FaceBMIDataset(
        csv_path="data/data.csv",
        images_dir="data/Images",
        is_training_value=1,
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# --- 2) Loss function and optimizer ---
# MSELoss = Mean Squared Error: averages (prediction - truth) squared.
# It is the standard "scoring rule" for regression (predicting a number).
loss_function = nn.MSELoss()

# Adam = a popular optimizer (the "coach" that adjusts weights).
# lr = learning rate = how big each adjustment step is. Small, because fine-tuning.
optimizer = Adam(model.parameters(), lr=1e-4)


# --- 3) The Training Loop ---
num_epochs = 50      # How many times we go through the whole dataset.

for epoch in range(num_epochs):
    model.train()           # Put the model in "training mode".
    running_loss = 0.0      # initialize: to track the average loss this epoch.
    num_batches = 0 


    for images, bmis in train_loader:
        # Move this batch onto the device (same place as the model).
        images = images.to(device)
        bmis = bmis.to(device)

        # Filter out invalid samples (no face found -> bmi was set to -1).
        valid = bmis > 0        # [True, True, False, ...] Masking 
        if valid.sum() == 0:    # True = 1, False = 0 | If all False == 0. then pass this batch!
            continue
        images = images[valid]  # Masking
        bmis = bmis[valid]      # Masking


        # === The Core 4-step Cycle === #

        # (a) forward: push images through the model to get predictions.
        predictions = model(images)             # shape (batch, 1)
        predictions = predictions.squeeze(1)    # shape (batch, ) to match bmis | erase dimension

        # (b) score: how wrong are we? (one number)
        loss = loss_function(predictions, bmis)

        # (c) reset old gradients, then backward: compute how to fix each weight.
        optimizer.zero_grad()  # reset gradients
        loss.backward()        # back propagation

        # (d) update: the optimizer nudges the weights to reduce loss.
        optimizer.step()

        # Book keeping for the average loss.
        running_loss += loss.item()
        num_batches += 1


    average_loss = running_loss / num_batches
    print(f"epoch {epoch+1} / {num_epochs} | average loss: {average_loss:.2f}")


# --- 4) Save the trained model ---
torch.save(model.state_dict(), "bmi_finetuned.pth")
print("saved model to bmi_finetuned.pth")


        
