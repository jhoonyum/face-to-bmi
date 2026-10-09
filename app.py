# app.py
# Goal: a Streamlit web app that predicts BMI from a webcam photo or uploaded image.
# It uses the best model: ArcFace embeddings + SVR (saved in arcface_svr_model.pkl).
#
# How to run:
#   streamlit run app.py
# Then open the URL it prints (usually http://localhost:8501) in a browser.

import streamlit as st          # Streamlit: turns Python into a web app
import numpy as np
import cv2                       # OpenCV: reads images as numpy arrays (BGR)
import joblib                    # to load the saved scikit-learn model
from insightface.app import FaceAnalysis   # the ArcFace detect + embed tool
from PIL import Image            # to handle uploaded/camera images

# --- Page setup ---
st.set_page_config(page_title="Face-to-BMI", page_icon="📷")
st.title("Face-to-BMI Predictor")
st.write("Take a photo or upload an image to estimate BMI from a face.")
st.caption("Model: ArcFace embeddings + SVR (Pearson r = 0.69 on test set)")

# --- Load the model and the face analyzer ONCE ---
# @st.cache_resource tells Streamlit: "build this only once, then reuse it."
# Without it, the heavy model would reload on every click (very slow).
@st.cache_resource
def load_tools():
    # Load the trained ArcFace + SVR pipeline from the saved file.
    model = joblib.load("arcface_svr_model.pkl")

    # Set up the ArcFace face analyzer (detection + embedding), on CPU.
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

    return model, face_app

model, face_app = load_tools()

# --- The prediction function ---
def predict_bmi_from_image(pil_image):
    # Convert the PIL image to a numpy array in BGR order (what InsightFace expects).
    # PIL gives RGB; cv2.cvtColor flips RGB -> BGR.
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Detect + align + embed the face(s). Returns a list of faces (maybe empty).
    faces = face_app.get(image_bgr)
    if len(faces) == 0:
        return None   # no face found

    # If several faces, keep the most confident one (highest detection score).
    face = max(faces, key=lambda f: f.det_score)

    # Get the 512-number ArcFace embedding for this face.
    embedding = face.embedding.reshape(1, -1)   # shape (1, 512) for the model

    # The pipeline scales the embedding and predicts BMI in one step.
    bmi = model.predict(embedding)[0]
    return bmi

# --- BMI category helper (for a friendly label) ---
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# --- Let the user choose: webcam or upload ---
input_mode = st.radio("Choose input method:", ["Webcam", "Upload image"])

image = None
if input_mode == "Webcam":
    # st.camera_input shows a live camera widget in the browser.
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        image = Image.open(camera_photo)
else:
    # st.file_uploader lets the user pick an image file.
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        image = Image.open(uploaded)

# --- Run prediction when we have an image ---
if image is not None:
    st.image(image, caption="Input image", width=300)

    with st.spinner("Analyzing face..."):
        bmi = predict_bmi_from_image(image)

    if bmi is None:
        st.error("No face detected. Please try a clearer, front-facing photo.")
    else:
        category = bmi_category(bmi)
        # Show the result as a big number with its category.
        st.metric(label="Predicted BMI", value=f"{bmi:.1f}", delta=category, delta_color="off")

        # Honest disclaimer (important for the ethics part of the project).
        st.warning(
            "This is an experimental estimate from a face photo only. "
            "It is not a medical measurement and can be inaccurate for any individual. "
            "Do not use it for health decisions."
        )
