# 12_dataset.py
# Goal: define a Dataset that gives (face_tensor, bmi) pairs,
# and wrap it in a DataLoader that serves shuffled batches.

import os
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN
from PIL import Image

class FaceBMIDataset(Dataset):
    # __init__ prepares the list of (image_name, bmi) we will serve.
    def __init__(self, csv_path, images_dir, is_training_value):
        data_frame = pd.read_csv(csv_path, index_col=0)
        
        # Keep only rows for this split (1=train, 0=test).
        data_frame = data_frame[data_frame["is_training"] == is_training_value]

        self.images_dir = images_dir

        # Store the rows we will use as a simple list of (name, bmi)
        self.samples = list(zip(data_frame["name"], data_frame["bmi"]))    # Data Samples

        # The face detector.
        # On CPU because MTCNN breaks on mps.
        # It crops/aligns each face to 160 x 160 and returns a tensor 
        self.face_detector = MTCNN(image_size=160, device="cpu")

    
    # __len__ tells PyTorch how many samples exist.
    def __len__(self):
        return len(self.samples)

    # __getitem__ returns the i-th (face_tensor, bmi).
    def __getitem__(self, index):
        image_name, bmi = self.samples[index]
        image_path = os.path.join(self.images_dir, image_name)

        # Some rows in the CSV have no matching image file on disk (244 of them).
        # If the file is missing, return a zero tensor + bmi=-1 (filtered out in training).
        if not os.path.exists(image_path):
            return torch.zeros(3, 160, 160), torch.tensor(-1.0, dtype=torch.float32)

        # Open and detect the face -> tensor of shape (3, 160, 160), or None.
        image = Image.open(image_path).convert("RGB")
        face_tensor = self.face_detector(image)

        # If no face found, return a zero tensor (we will filter these out later).
        if face_tensor is None:
            face_tensor = torch.zeros(3, 160, 160)
            bmi = -1.0      # mark as invalid

        # Return the face tensor and the bmi as a float tensor.
        return face_tensor, torch.tensor(bmi, dtype=torch.float32)


# Quick test: build the train dataset and a loader, pull ONE batch.
if __name__ == "__main__":
    dataset = FaceBMIDataset(
            csv_path="data/data.csv",
            images_dir="data/Images",
            is_training_value=1,
    )
    print("dataset size:", len(dataset))


    # DataLoader serves batches. 
    # batch_size=16 : 16 images at a time.
    # shuffle=True : randomize order each epoch (help training).
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Pull one batch and check shapes.
    images, bmis = next(iter(loader))
    print("one batch images:", images.shape)    # expect (16, 3, 160, 160)
    print("one batch bmis:", bmis.shape)        # expect (16, )


