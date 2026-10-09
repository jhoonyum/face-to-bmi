# 20_extract_finetuned_embeddings.py
# 파인튜닝된 FaceNet(v2) 모델에서 head를 떼고 backbone 출력(512차원)만 뽑아 저장.
# 이 임베딩은 BMI 특화 학습을 거쳤으므로 frozen 임베딩과 다른 정보를 담을 수 있다.

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from importlib import import_module

BMIModel = import_module("11_model").BMIModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---- 1) 파인튜닝 모델 불러오기 ----
model = BMIModel().to(device)
model.load_state_dict(torch.load("bmi_finetuned_v2.pth", map_location=device))
model.eval()

# ---- 2) head를 떼고 backbone만 사용 ----
# forward 대신 backbone만 직접 호출해서 512차원 임베딩을 얻는다.
backbone = model.backbone
backbone.eval()

# ---- 3) 얼굴 검출기 ----
face_detector = MTCNN(image_size=160, device="cpu")

# ---- 4) 라벨 파일 읽기 ----
import pandas as pd
df = pd.read_csv("data/data.csv", index_col=0)
print(f"total records: {len(df)}")

# ---- 5) 전체 이미지에서 임베딩 추출 ----
import os
embeddings  = []
bmi_list    = []
is_train_list = []
name_list   = []
skipped     = 0

for i, row in df.iterrows():
    img_path = os.path.join("data/Images", row["name"])

    # 파일 없으면 건너뜀
    if not os.path.exists(img_path):
        skipped += 1
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        face_tensor = face_detector(img)
        if face_tensor is None:
            skipped += 1
            continue

        # (3,160,160) -> (1,3,160,160) 배치로 포장
        face_batch = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = backbone(face_batch)   # (1, 512)

        embeddings.append(emb.squeeze(0).cpu().numpy())
        bmi_list.append(row["bmi"])
        is_train_list.append(row["is_training"])
        name_list.append(row["name"])

    except Exception as e:
        skipped += 1
        continue

    if (len(embeddings)) % 200 == 0:
        print(f"  processed {len(embeddings)} faces...")

print(f"\ndone. extracted: {len(embeddings)}, skipped: {skipped}")

# ---- 6) 저장 ----
X           = np.array(embeddings)
y           = np.array(bmi_list)
is_training = np.array(is_train_list)
names       = np.array(name_list)

np.savez("features/finetuned_v2_embeddings.npz",
         X=X, y=y, is_training=is_training, names=names)
print(f"saved to features/finetuned_v2_embeddings.npz  shape: {X.shape}")
