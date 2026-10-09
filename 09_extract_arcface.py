# 09_extract_arcface.py
# Goal: same as 04_extract_all.py, but use ArcFace (InsightFace) for embeddings.
# Flow: for each row -> detect face -> embedding -> stack -> save.

import os
import time
import numpy as np 
import pandas as pd 
import cv2                                  # InsightFace reads images via OpenCV (numpy arrays).
from insightface.app import FaceAnalysis    # the all-in-one detect + embed tool.

# --- 1) Set up the ArcFace model bundle ---
# name = "buffalo_l" is a model pack: a face detector + the ArcFace embedding model.
# It auto-downloads the first time. providers=["CPUExecutionProvider"] runs on CPU 
# (onnxruntime on Mac; this is the most reliable choice).
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

# prepare() loads the models.
# det_size is the size used for face detection.
# ctx_id = -1 means "use CPU"
# a GPU id like 0 means GPU, but we use CPU here.
app.prepare(ctx_id=-1, det_size=(640, 640))
print("ArcFace ready.")

# --- 2) Read the label table ---
data_frame = pd.read_csv("data/data.csv", index_col=0)
print("labels:", data_frame.shape)

# --- 3) Lists that grow together, in the same order (the matching rule) ---
embedding_list = []
bmi_list = []
gender_list = []
is_training_list = []
name_list = []

skipped_missing_file = 0
skipped_no_face = 0

start_time = time.time()

for row in data_frame.itertuples(index=False):
    image_name = row.name
    image_path = os.path.join("data/Images", image_name)

    if not os.path.exists(image_path):
        skipped_missing_file += 1
        continue

    # Read the image as a numpy array 
    # BGR color order, which is what OpenCV/InsightFace use.
    image = cv2.imread(image_path)
    if image is None:
        skipped_missing_file += 1
        continue

    # Detect + align + embed, all in one call.
    # Returns a list of faces (could be empty)
    faces = app.get(image)
    if len(faces) == 0:
        skipped_no_face += 1
        continue

    # If several faces are found, keep the most confident one.
    # Highest det_score!
    face = max(faces, key=lambda f: f.det_score)

    # The ArcFace embedding: 512 numbers describing this face.
    embedding_numbers = face.embedding

    # Stack everything in the same order (so embedding[i] matches bmi[i], etc.)
    embedding_list.append(embedding_numbers)
    bmi_list.append(float(row.bmi))
    gender_list.append(1 if str(row.gender).lower().startswith("m") else 0)
    is_training_list.append(int(row.is_training))
    name_list.append(image_name)

    done = len(name_list) + skipped_missing_file + skipped_no_face
    if done % 500 == 0:
        print(f"    processed {done} /  {len(data_frame)} ...")


# --- 4) Turn lists into arrays and save ---
X = np.vstack(embedding_list)
y = np.array(bmi_list, dtype=np.float32)
gender = np.array(gender_list, dtype=np.int8)
is_training = np.array(is_training_list, dtype=np.int8)
names = np.array(name_list)

os.makedirs("features", exist_ok=True)
np.savez_compressed(
        "features/arcface_embeddings.npz",
        X=X, y=y, gender=gender, is_training=is_training, names=names,
)

print("\n--- done ---")
print("X shape:", X.shape)
print("skipped (file missing):", skipped_missing_file)
print("skipped (no face):", skipped_no_face)
print(f"time: {time.time() - start_time:.1f} seconds")
print("saved to: features/arcface_embeddings.npz")

