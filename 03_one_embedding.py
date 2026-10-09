# 03_one_embedding.py
# Goal: turn ONE face image into one embedding vector (512 numbers).

import torch    # PyTorch: the deep learning engine.
from facenet_pytorch import MTCNN, InceptionResnetV1 
# face detector + embedding model
from PIL import Image 
# Pillow: opens image files

# Pick the device. 
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# MTCNN finds the face in a photo and crops/aligns it to 160 x 160.
# image_size = 160 is what the embedding model expects as input.
face_detector = MTCNN(image_size=160, device="cpu")

# InceptionResnetV1 turns an aligned face into a 512-number embedding.
# pretrained="vggface2" loads weights already trained on millions of faces.
# .eval() puts the model in "use mode" (not learning mode).
# .to(device) moves the model onto our chosen device (mps or cpu).
embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Open one image file. PIL reads it; convert("RGB") makes sure it has 3 color channels.
image_path = "data/Images/img_0.bmp"
image = Image.open(image_path).convert("RGB")

# Step 1: detect + crop + align the face.
# Result is a tensor (a number grid) or None.
face_tensor = face_detector(image)

if face_tensor is None:
    print("No face detected in", image_path)
else:
    # Step 2: make the embedding.
    # face_tensor.unsqueeze(0) adds a "batch" dimension 
    # (the model expects a stack of images, even if we only have one).
    # torch.no_grad() means "we are only using the model, not training it" -> faster.
    with torch.no_grad():
        embedding = embedding_model(face_tensor.unsqueeze(0).to(device))
    
    # embedding is a tensor of shape(1, 512). 
    # Print its shape and the first 5 numbers.
    print("embedding shape:", embedding.shape)
    print("first 5 numbers:", embedding[0][:5])

