# 04_extract_all.py
# Goal: turn ALL face images into embeddings, and save them to one file.
#
# Big picture (the flow we drew):
#   for each row in the label table:
#       find the image -> detect face -> make embedding ->
#       stack the embedding together with its bmi/gender/etc (same order)
#   then: save everything into one .npz file.

import os                   # to build file paths and check files
import time                 # to measure how long extraction takes
import numpy as np          # to stack numbers and save the .npz file
import pandas as pd         # to read the label CSV.
import torch                # the deep learning engine
from facenet_pytorch import MTCNN, InceptionResnetV1 # face detector + embedding model
from PIL import Image       # to open image files

# Pick device: Apple GPU (mps) for the heavy embedding model, cpu for MTCNN.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# Face detector on CPU (it has an op that breaks on mps)
face_detector = MTCNN(image_size=160, device="cpu")

# Embedding model on the chosen device, in "use mode".
embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Read the label table.
# index_col=0 drops the "Unnamed: 0" row-number column.
data_frame = pd.read_csv("data/data.csv", index_col=0)
print("labels:", data_frame.shape)


# These lists will grow as we process each image.
# They stay in the SAME order.
# So, embedding[i] always matches bmi[i], gender[i], etc. (no mismatch).
embedding_list = []     # each item: 512 numbers
bmi_list = []           # each item: one BMI number
gender_list = []        # each item: 1 for male, 0 for female
is_training_list = []   # each item: 1 for train, 0 for test
name_list = []          # each item: the image file name 

skipped_missing_file = 0    # count images listed in CSV but not found on disk.
skipped_no_face = 0         # count images where MTCNN found no face.

start_time = time.time()

# Go through the label table one row at a time.
# itertuples() gives us each row: we read row.name_? carefully below.
for row in data_frame.itertuples(index=False):
    image_name = row.name 
    image_path = os.path.join("data/Images", image_name)

    # Skip if the image file does not exist on disk.
    if not os.path.exists(image_path):
        skipped_missing_file += 1 
        continue

    # Open the image and force it to RGB (3 channels)
    image = Image.open(image_path).convert("RGB")

    # Detect + crop + align the face. Returns a tensor, or None if no face found.
    face_tensor = face_detector(image)
    if face_tensor is None:
        skipped_no_face += 1 
        continue

    # Make the embedding, no_grad = use only, not training (faster, less memory)
    with torch.no_grad():
        embedding = embedding_model(face_tensor.unsqueeze(0).to(device))

    # Move the embedding back to cpu and turn it into plain numbers (512 of them).
    embedding_numbers = embedding[0].cpu().numpy()

    # Stack this row's results together, in the same order, so they stay matched.
    embedding_list.append(embedding_numbers)
    bmi_list.append(float(row.bmi))
    gender_list.append(1 if str(row.gender).lower().startswith("m") else 0)
    is_training_list.append(int(row.is_training))
    name_list.append(image_name)

    # Print progress every 500 images so we know it is alive.
    done = len(name_list) + skipped_missing_file + skipped_no_face
    if done % 500 == 0:
        print(f" processed {done} / {len(data_frame)} ...")


# Turn the lists into numpy arrays (a grid of numbers, easy to save and load).
X = np.vstack(embedding_list)               # shape: (num_faces, 512)
y = np.array(bmi_list, dtype=np.float32)    # shape: (num_faces, )
gender = np.array(gender_list, dtype=np.int8)
is_training = np.array(is_training_list, dtype=np.int8)
names = np.array(name_list)

# Make a folder to keep the saved features, if it does not exist yet.
os.makedirs("features", exist_ok=True)

# Save everything into ONE file. We can load it back instantly later.
np.savez_compressed(
        "features/facenet_embeddings.npz",
        X=X, y=y, gender=gender, is_training=is_training, names=names,
)

# Report what happened.
print("\n--- done ---")
print("X shape:", X.shape) # (num_faces, 512)
print("skipped (file missing):", skipped_missing_file)
print("skipped (no face):", skipped_no_face)
print(f"time: {time.time() - start_time:.1f} seconds")
print("Saved to: features/facenet_embeddings.npz")
