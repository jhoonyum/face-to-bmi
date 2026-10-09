# 24_extract_topofr.py
# TopoFR (Glint360K R100) 임베딩 추출.
# 입력: 112x112 RGB, [-1,1] 정규화. forward는 net(img, phase='infer').
# InsightFace 정렬을 재활용한다.

import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2

# ---- 1) TopoFR 코드를 import 경로에 추가 ----
TOPOFR_DIR = os.path.abspath("third_party/TopoFR")
sys.path.insert(0, TOPOFR_DIR)

from backbones import get_model

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---- 2) 모델 로드 ----
WEIGHT = os.path.abspath(
    "third_party/TopoFR/work_dirs/glint360k_r100/Glint360K_R100_TopoFR_9760.pt"
)
ckpt = torch.load(WEIGHT, map_location="cpu")
num_classes = ckpt["weight"].shape[0]
print("num_classes:", num_classes)

net = get_model("r100", fp16=False, num_classes=num_classes)
net.load_state_dict(ckpt)
net = net.to(device).eval()
print("TopoFR R100 loaded.")

# ---- 3) InsightFace 정렬 ----
from insightface.app import FaceAnalysis
from insightface.utils import face_align

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def get_topofr_embedding(image_path):
    img = cv2.imread(image_path)   # BGR
    if img is None:
        return None
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f.det_score)

    # 112x112 정렬 (BGR)
    aligned_bgr = face_align.norm_crop(img, landmark=face.kps, image_size=112)

    # TopoFR inference.py와 동일: BGR -> RGB, [-1,1] 정규화
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(aligned_rgb).permute(2, 0, 1).unsqueeze(0).float()
    t.div_(255).sub_(0.5).div_(0.5)
    t = t.to(device)

    with torch.no_grad():
        feat = net(t, phase="infer")
    return feat.squeeze(0).cpu().numpy()

# ---- 4) 전체 추출 ----
df = pd.read_csv("data/data.csv", index_col=0)
embeddings, bmis, is_train, names = [], [], [], []
skipped = 0

for i, row in df.iterrows():
    path = os.path.join("data/Images", row["name"])
    if not os.path.exists(path):
        skipped += 1
        continue
    emb = get_topofr_embedding(path)
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

np.savez("features/topofr_embeddings.npz",
         X=X, y=np.array(bmis), is_training=np.array(is_train), names=np.array(names))
print("saved to features/topofr_embeddings.npz")
