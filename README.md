Samajh gaya MD 👌 — tum apne **helmet detection project** ke liye ek **complete README.md file** chahte ho jo professional aur full documentation style me ho. Main tumhe ek ready‑to‑use draft de raha hoon jo GitHub pe directly use kiya ja sakta hai.

---

# 🪖 Helmet Detection using YOLOv8

## 📌 Overview

This project implements a **helmet detection system** using **YOLOv8**. It supports:

- **Image detection** (bounding boxes drawn on uploaded images)
- **Video detection** (live frame‑by‑frame detection or batch annotated video output)
- **APIs** via **FastAPI** and **Flask**
- **Interactive UI** via **Streamlit**

The goal is to detect whether a person is **With Helmet** or **Without Helmet** in real‑time or batch mode.

---

## 📂 Project Structure

```
helmet-detection/
│
├── app/                      # FastAPI app or UI logic
├── artifacts/                # Saved model artifacts or intermediate files
├── dataset/                  # Training and validation data
├── flowcharts/               # Visual diagrams for architecture or workflow
├── logs/                     # Logging outputs
├── notebooks/                # Jupyter notebooks for experimentation
├── outputs/                  # Annotated image/video outputs
├── runs/                     # YOLO training runs
├── src/                      # Core source code
│   ├── components/           # Reusable modules
│   ├── constant/             # Constants like class names, paths
│   ├── entity/               # Data schemas or config entities
│   ├── ml/                   # Model training, evaluation logic
│   ├── pipeline/             # Prediction and training pipelines
│   └── utils/                # Logger, exception, helper functions
│       ├── exception.py
│       └── logger.py
│
├── test/                     # Unit tests
├── TrainedModel/             # Final trained YOLOv8 model (e.g., best.pt)
│
├── .gitignore                # Ignore temp, logs, venv, etc.
├── app_streamlit.py          # Streamlit app for image/video detection
├── Dockerfile                # Containerization setup
├── LICENSE                   # Project license
├── README.md                 # Documentation
├── requirements.txt          # Python dependencies

```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/helmet-detection.git
cd helmet-detection
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash/Unix
# OR
.venv\Scripts\activate          # Windows CMD/PowerShell
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Streamlit App (UI)

Run:

```bash
streamlit run app_streamlit.py
```

- Opens at `http://localhost:8501`
- Upload image → annotated image shown instantly
- Upload video → live detection frame‑by‑frame

---

### 🔹 FastAPI (REST API)

Run:

```bash
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints:

- `POST /predict-image/` → Upload image → JSON + annotated file saved
- `POST /predict-video/` → Upload video → JSON + annotated file saved

---

### 🔹 Flask App (REST API)

Run:

```bash
python app_flask.py
```

Endpoints:

- `POST /predict-image`
- `POST /predict-video`

---

## 📊 Example Response (Image)

```json
{
  "message": "Image processed successfully",
  "predictions": [
    {
      "class_id": 0,
      "class_name": "With Helmet",
      "confidence": 0.83,
      "bbox_xyxy": [12, 34, 56, 78]
    },
    {
      "class_id": 1,
      "class_name": "Without Helmet",
      "confidence": 0.67,
      "bbox_xyxy": [90, 120, 150, 200]
    }
  ],
  "visual_output_dir": "outputs/images"
}
```

---

## 🛠 Requirements

- Python 3.9+
- Streamlit
- FastAPI
- Flask
- Uvicorn
- OpenCV
- Ultralytics YOLOv8

All dependencies are listed in `requirements.txt`.

---

## 📌 Features

- ✅ Real‑time image detection (Streamlit)
- ✅ Live video detection (Streamlit)
- ✅ REST APIs (FastAPI + Flask)
- ✅ Annotated outputs saved automatically
- ✅ Modular pipeline with logging + exception handling

---

## 📜 License

This project is licensed under the MIT License.

---
