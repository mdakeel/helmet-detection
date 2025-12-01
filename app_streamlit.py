import streamlit as st
import cv2
import os
from ultralytics import YOLO
from src.constant import TRAINED_MODEL_DIR, TRAINED_MODEL_NAME

# Load YOLO model
model_path = os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
model = YOLO(model_path)

st.title("🪖 Helmet Detection App (YOLOv8)")
st.write("Upload an image or video to detect helmets in real-time.")

# ---------------- IMAGE DETECTION ----------------
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    temp_path = f"temp_{uploaded_image.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_image.read())

    # Run YOLO detection directly on image
    results = model(temp_path)

    # Annotated image (YOLO returns numpy array with bounding boxes)
    annotated_img = results[0].plot()

    # Show annotated image instantly
    st.image(annotated_img, caption="Detected Helmets", channels="BGR", use_column_width=True)

    os.remove(temp_path)

# ---------------- VIDEO DETECTION ----------------
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    temp_path = f"temp_{uploaded_video.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_path)
    stframe = st.empty()  # placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection on current frame
        results = model(frame)
        annotated_frame = results[0].plot()

        # Show frame in Streamlit
        stframe.image(annotated_frame, channels="BGR")

    cap.release()
    os.remove(temp_path)
