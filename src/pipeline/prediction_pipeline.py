import os
import sys
from ultralytics import YOLO
from src.logger import logging
from src.exception import CustomException
from src.constant import TRAINED_MODEL_DIR, TRAINED_MODEL_NAME, PREDICTION_CLASSES


class PredictionPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self.model = YOLO(self.model_path)
            logging.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict_image(self, image_path: str):
        """
        Run prediction on a single image and return bounding boxes + class info.
        """
        try:
            results = self.model.predict(source=image_path, verbose=False)
            predictions = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    predictions.append({
                        "class_id": cls_id,
                        "class_name": PREDICTION_CLASSES[cls_id] if cls_id < len(PREDICTION_CLASSES) else str(cls_id),
                        "confidence": conf,
                        "bbox_xyxy": xyxy
                    })
            return predictions
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict_video(self, video_path: str, save_output: bool = True):
        """
        Run prediction on a video and optionally save annotated output.
        """
        try:
            results = self.model.predict(source=video_path, stream=True, save=save_output,
                             project="outputs", name="videos", verbose=False)
            predictions = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    predictions.append({
                        "class_id": cls_id,
                        "class_name": PREDICTION_CLASSES[cls_id] if cls_id < len(PREDICTION_CLASSES) else str(cls_id),
                        "confidence": conf,
                        "bbox_xyxy": xyxy
                    })
            return predictions
        except Exception as e:
            raise CustomException(e, sys) from e
