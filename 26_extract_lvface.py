# 26_extract_lvface.py
# LVFace (ViT-B, Glint360K) 임베딩 추출 (ONNX).
# inference_onnx.py의 LVFaceONNXInferencer를 재활용.
# 입력: 112x112 RGB, [-1,1]. InsightFace 정렬 사용.

import os
import sys
import numpy as np
import pandas as pd
import cv2

# ---- 1) LVFace 추론 클래스 가져오기 ----
LVFACE_DIR = os.path.abspath("third_party/LVFace")
sys.path.insert(0, LVFACE_DIR)
from inference_onnx import LVFaceONNXInferencer

MODEL_PATH = os.path.abspath(
    "third_party/LVFace/LVFace_model/LVFace-B_Glint360K/LVFace-B_Glint360K.onnx"
)

# use_gpu=False : 맥에서는 CPU로 (onnxruntime CPU 버전).
inferencer = LVFaceONNXInferencer(model_path=MODEL_PATH, use_gpu=False)
print("LVFace ViT-B loaded.")

# ---- 2) InsightFace 정렬 ----
from insightface.app import FaceAnalysis
from insightface.utils import face_align

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def get_lvface_embedding(image_path):
    img = cv2.imread(image_path)   # BGR
    if img is None:
        return None
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f.det_score)

    # 112x112 정렬 (BGR) -> LVFace 전처리기는 BGR 입력을 기대 (내부에서 RGB 변환)
    aligned_bgr = face_align.norm_crop(img, landmark=face.kps, image_size=112)

    # LVFace의 _preprocess_image는 BGR을 받아 내부에서 RGB 변환 + 정규화.
    img_tensor = inferencer._preprocess_image(aligned_bgr)
    output = inferencer.ort_session.run(
        [inferencer.output_name],
        {inferencer.input_name: img_tensor}
    )
    return np.ravel(output[0])

# ---- 3) 전체 추출 ----
df = pd.read_csv("data/data.csv", index_col=0)
embeddings, bmis, is_train, names = [], [], [], []
skipped = 0

for i, row in df.iterrows():
    path = os.path.join("data/Images", row["name"])
    if not os.path.exists(path):
        skipped += 1
        continue
    try:
        emb = get_lvface_embedding(path)
    except Exception:
        emb = None
    if emb is None:
        skipped += 1
        continue
    embeddings.append(emb)
    bmis.append(row["bmi"])
    is_train.append(row["is_training"])
    names.append(row["name"])
    if len(embeddings) % 200 == 0:
        print(f"  processed {len(embeddings)}...")

X = np.array(embeddings)
print(f"\ndone. extracted: {len(embeddings)}, skipped: {skipped}, shape: {X.shape}")

np.savez("features/lvface_embeddings.npz",
         X=X, y=np.array(bmis), is_training=np.array(is_train), names=np.array(names))
print("saved to features/lvface_embeddings.npz")
