# 14_predict.py
# Goal: load the trained model and predict BMI for ONE face image.
# This is "inference": using a trained model to answer, not training it.

import torch
from facenet_pytorch import MTCNN
from PIL import Image


# Bring in the model class we defined in 11_model.py.
# We need the same class shape to load the saved weights into it.
from importlib import import_module
BMIModel = import_module("11_model").BMIModel

# --- 1) Set up device, model, and face detector (done once) ---
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Build an empty model with the same structure as the one we trained.
model = BMIModel().to(device)

# Load the weights we saved during training into this empty model.
model.load_state_dict(torch.load("bmi_finetuned.pth", map_location=device))

# Put the model in "use mode". 
# Turn off training-only behaviors.
model.eval()

# The face detector, on CPU.
face_detector = MTCNN(image_size=160, device="cpu")

# --- 2) The prediction function ---
def predict_bmi(image_path):
    # Open the image and force RGB (3 color channels).
    image = Image.open(image_path).convert("RGB")

    # Detect + crop + align the face -> tensor (3, 160, 160), None if no face.
    face_tensor = face_detector(image)
    if face_tensor is None:
        return None         # Tell the caller "no face found"
    

    # Wrap the single face into a batch of 1: (3, 160, 160) -> (1, 3, 160, 160).
    face_batch = face_tensor.unsqueeze(0).to(device)

    # Run the model WITHOUT tracking gradients (we are using, not learning).
    with torch.no_grad():
        prediction = model(face_batch)      # shape(1, 1)

    # Pull out the single number and return it as a plain Python float.
    bmi = prediction.item()
    return bmi

# --- 3) Quick test: predict on one image and print the result ---
if __name__ == "__main__":
    test_image = "data/Images/img_0.bmp"
    result = predict_bmi(test_image)

    if result is None:
        print("no face detected in", test_image)
    else:
        print(f"predicted BMI for {test_image}: {result:.1f}")
        