# 13_train_v3.py
# v2에서 데이터 증강(augmentation)만 추가한 버전.
# 학습 중 배치마다 이미지를 랜덤 변형해서 과적합을 늦추고 더 오래 학습.

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from importlib import import_module

BMIModel      = import_module("11_model").BMIModel
FaceBMIDataset = import_module("12_dataset").FaceBMIDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---- 1) BMI 정규화 (v2와 동일) ----
df       = pd.read_csv("data/data.csv", index_col=0)
train_df = df[df["is_training"] == 1]
bmi_mean = float(train_df["bmi"].mean())
bmi_std  = float(train_df["bmi"].std())
print(f"BMI mean: {bmi_mean:.2f}  std: {bmi_std:.2f}")

# ---- 2) 증강 함수 ----
# MTCNN이 돌려주는 텐서는 [-1, 1] 범위.
# TF 함수들은 [0, 1]을 기대해서, 변환 전후로 범위를 조정.
# 증강은 학습 때만 씀. 평가(evaluate)에선 쓰지 않음.
def augment_batch(images):
    """
    images: (B, 3, 160, 160) tensor, range [-1, 1]
    returns: same shape, randomly augmented
    """
    result = []
    for img in images:
        # [-1,1] -> [0,1]
        img = (img + 1) / 2

        # 좌우 반전 (얼굴은 좌우 대칭, BMI와 무관)
        if random.random() > 0.5:
            img = TF.hflip(img)

        # 약간의 회전 (최대 15도, 얼굴이 약간 기울 수 있으니)
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle)

        # 밝기/대비 변화 (BMI는 조명과 무관)
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        # 범위 벗어나지 않게 클램프
        img = torch.clamp(img, 0, 1)

        # [0,1] -> [-1,1] 로 되돌리기
        img = img * 2 - 1
        result.append(img)
    return torch.stack(result)

# ---- 3) 모델 구성 + 부분 해동 (v2와 동일) ----
model = BMIModel().to(device)

for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

UNFREEZE_BACKBONE = ["block8", "avgpool_1a", "last_linear", "last_bn"]
for name, param in model.backbone.named_parameters():
    if any(layer in name for layer in UNFREEZE_BACKBONE):
        param.requires_grad = True

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")

# ---- 4) 옵티마이저 (v2와 동일) ----
optimizer = torch.optim.Adam([
    {"params": model.head.parameters(),
     "lr": 1e-3},
    {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad],
     "lr": 1e-5},
])
criterion = nn.MSELoss()

# ---- 5) 데이터 로더 ----
train_loader = DataLoader(
    FaceBMIDataset("data/data.csv", "data/Images", is_training_value=1),
    batch_size=16, shuffle=True,
)
test_loader = DataLoader(
    FaceBMIDataset("data/data.csv", "data/Images", is_training_value=0),
    batch_size=16, shuffle=False,
)

# ---- 6) 평가 함수 (v2와 동일, 증강 없음) ----
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
            out     = model(images).squeeze(1)
            out_bmi = out * bmi_std + bmi_mean
            preds.extend(out_bmi.cpu().numpy())
            truths.extend(bmis.cpu().numpy())
    r   = pearsonr(np.array(truths), np.array(preds))[0]
    mae = mean_absolute_error(np.array(truths), np.array(preds))
    return r, mae

# ---- 7) 학습 루프 ----
num_epochs = 100
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

        # 증강 적용 (학습할 때만)
        images = augment_batch(images)

        bmis_norm = (bmis - bmi_mean) / bmi_std

        optimizer.zero_grad()
        predictions = model(images).squeeze(1)
        loss = criterion(predictions, bmis_norm)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if epoch % 5 == 0 or epoch == 1:
        r, mae = evaluate(test_loader)
        tag = "  <-- best" if r > best_r else ""
        if r > best_r:
            best_r     = r
            best_epoch = epoch
            torch.save(model.state_dict(), "bmi_finetuned_v3.pth")
        print(f"epoch {epoch:3d}/{num_epochs} | loss: {avg_loss:.4f} | test r: {r:.3f} | MAE: {mae:.2f}{tag}")
    else:
        print(f"epoch {epoch:3d}/{num_epochs} | loss: {avg_loss:.4f}")

print(f"\nbest test r = {best_r:.3f}  at epoch {best_epoch}")
print("saved to bmi_finetuned_v3.pth")
