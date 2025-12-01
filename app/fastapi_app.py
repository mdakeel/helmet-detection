# src/fastapi_app.py
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from src.constant import TRAINED_MODEL_DIR, TRAINED_MODEL_NAME

app = FastAPI(title="Helmet Detection API", description="YOLOv8 Inference API")

# Load trained model once at startup
model_path = os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = YOLO(model_path)

def extract_predictions(results):
    preds = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            preds.append({
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": conf,
                "bbox_xyxy": xyxy
            })
    return preds

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    results = model.predict(source=temp_path, verbose=False)
    os.remove(temp_path)
    return {"predictions": extract_predictions(results)}

@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    # Save visualization outputs to outputs/videos/
    os.makedirs("outputs/videos", exist_ok=True)
    results = model.predict(source=temp_path, save=True, project="outputs", name="videos", verbose=False)
    os.remove(temp_path)
    return {
        "message": "Video processed",
        "visual_output_dir": os.path.join("outputs", "videos"),
        "predictions_sample": extract_predictions(results)[:10]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
