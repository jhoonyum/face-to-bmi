# 11_model.py
# Goal: define a model = FaceNet backbone + a small BMI head (512 -> 64 -> 1)
# This file only DEFINES the model.
# We train it in the next steps.

import torch
import torch.nn as nn       # nn = neural network building blocks
from facenet_pytorch import InceptionResnetV1

class BMIModel(nn.Module):
    # __init__ declares the parts the model is made of.
    def __init__(self):
        super().__init__() # required boilerplate: set up the nn.Module machinery
        
        # The backbone: FaceNet. 
        # pretrained="vggface2" loads the learned weights.
        # This outputs a 512 number embedding for a face.
        self.backbone = InceptionResnetV1(pretrained="vggface2")

        # The head: turns 512 numbers into 1 BMI prediction.
        # Linear(512, 64): a layer with 512 inputs, 64 outputs (a weighted mix).
        # ReLU(): a non-linearity, lets the model learn curved relationship.
        # Linear(64, 1): final layer, 64 inputs -> 1 output (the BMI number).
        self.head = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )

    # forward defines how data flows through the parts.
    def forward(self, x):
        # x is a batch of face images, shape (batch, 3, 160, 160)
        embedding = self.backbone(x)    # -> (batch, 512)
        bmi = self.head(embedding)      # -> (batch, 1)
        return bmi

# Quick test: build the model and push one fake image through it.
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = BMIModel().to(device)
    print("model built on", device)

    # Make a fake batch of 2 images (random numbers), shape(2, 3, 160, 160).
    fake_images = torch.randn(2, 3, 160, 160).to(device)

    # Push them through. 
    # We expect output shape(2, 1): 2 images, 1 BMI each.
    output = model(fake_images)
    print("output shape:", output.shape)    # should be torch.Size([2, 1])
    

