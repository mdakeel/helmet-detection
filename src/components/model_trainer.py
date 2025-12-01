import os
import sys
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

from src.logger import logging
from src.exception import CustomException
from src.constant import * 
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifacts):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # Step 1: Load transformed datasets
            train_dataset = load_object(self.data_transformation_artifact.transformed_train_object)
            test_dataset  = load_object(self.data_transformation_artifact.transformed_test_object)

            train_loader = DataLoader(train_dataset,
                                      batch_size=self.model_trainer_config.BATCH_SIZE,
                                      shuffle=self.model_trainer_config.SHUFFLE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS)

            test_loader = DataLoader(test_dataset,
                                     batch_size=self.model_trainer_config.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=self.model_trainer_config.NUM_WORKERS)

            logging.info("Train and Test DataLoaders prepared")

            # Step 2: Fix for PyTorch 2.6 (allow YOLO DetectionModel class)
            #  Allow YOLO DetectionModel and Sequential for PyTorch 2.6
            torch.serialization.add_safe_globals([tasks.DetectionModel, torch.nn.modules.container.Sequential])

            # Step 3: Initialize YOLO model
            model = YOLO("yolov8n.pt")  # lightweight YOLOv8 model
            logging.info("YOLO model initialized")

            # Step 4: Train model
            data_yaml_path = os.path.join(os.getcwd(), "dataset", "data.yaml")

            model.train(
                data=data_yaml_path,
                epochs=self.model_trainer_config.EPOCH,
                batch=self.model_trainer_config.BATCH_SIZE,
                device=self.model_trainer_config.DEVICE
            )

            logging.info("Model training completed")

            # Step 5: Save trained model
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"Model saved at {self.model_trainer_config.TRAINED_MODEL_PATH}")

            # Step 6: Create artifact
            model_trainer_artifact = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
