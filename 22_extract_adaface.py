# 22_extract_adaface.py  (v2: transformers 우회, 순수 PyTorch 로드)
# CVLFace AdaFace IR101을 transformers 없이 직접 로드한다.
# PyTorch 2.2.2 그대로 작동. 입력: RGB 112x112, 정규화 mean=0.5 std=0.5.

import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import yaml
from omegaconf import OmegaConf

# ---- 1) CVLFace 폴더를 import 경로에 추가 ----
# 이 폴더 안의 models 패키지를 직접 쓰기 위해서.
ADAFACE_DIR = os.path.abspath("third_party/cvlface_adaface")
sys.path.insert(0, ADAFACE_DIR)

# get_model은 yaml_path의 상대경로(models/iresnet/...)를 기준으로 동작하므로
# 작업 디렉토리를 잠시 ADAFACE_DIR로 옮긴다.
original_cwd = os.getcwd()
os.chdir(ADAFACE_DIR)

from models import get_model   # transformers 거치지 않음

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---- 2) model.yaml 읽어서 모델 생성 + 가중치 로드 ----
with open("pretrained_model/model.yaml") as f:
    conf = OmegaConf.create(yaml.safe_load(f))

model = get_model(conf)                                  # IR101 생성
model.load_state_dict_from_path("pretrained_model/model.pt")   # 가중치 로드
model = model.to(device).eval()
print("AdaFace IR101 loaded (pure PyTorch).")

# 작업 디렉토리 원래대로 복귀
os.chdir(original_cwd)

# ---- 3) 얼굴 정렬: InsightFace로 112x112 정렬 크롭 얻기 ----
from insightface.app import FaceAnalysis
from insightface.utils import face_align

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def get_adaface_embedding(image_path):
    img = cv2.imread(image_path)   # BGR
    if img is None:
        return None
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f.det_score)

    # InsightFace 랜드마크로 112x112 정렬 (결과는 BGR)
    aligned_bgr = face_align.norm_crop(img, landmark=face.kps, image_size=112)

    # AdaFace는 RGB 입력 -> BGR을 RGB로 변환
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    # [0,255] -> [0,1] -> (x-0.5)/0.5 = [-1,1] 정규화
    t = torch.from_numpy(aligned_rgb).float() / 255.0   # (112,112,3)
    t = (t - 0.5) / 0.5
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)      # (1,3,112,112)

    with torch.no_grad():
        out = model(t)
        # CVLFace IResNetModel.forward는 임베딩 텐서를 반환.
        # 혹시 튜플이면 첫 요소가 임베딩.
        if isinstance(out, (tuple, list)):
            out = out[0]
    return out.squeeze(0).cpu().numpy()

# ---- 4) 전체 추출 ----
df = pd.read_csv("data/data.csv", index_col=0)
embeddings, bmis, is_train, names = [], [], [], []
skipped = 0

for i, row in df.iterrows():
    path = os.path.join("data/Images", row["name"])
    if not os.path.exists(path):
        skipped += 1
        continue
    emb = get_adaface_embedding(path)
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

np.savez("features/adaface_embeddings.npz",
         X=X, y=np.array(bmis), is_training=np.array(is_train), names=np.array(names))
print("saved to features/adaface_embeddings.npz")
