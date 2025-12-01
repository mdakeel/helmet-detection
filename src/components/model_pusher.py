
import os
import sys
import shutil
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts, ModelPusherArtifacts


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_trainer_artifact: ModelTrainerArtifacts):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            # Step 1: Check trained model path
            trained_model_path = self.model_trainer_artifact.trained_model_path
            if not os.path.exists(trained_model_path):
                raise FileNotFoundError(f"Trained model not found at {trained_model_path}")

            logging.info(f"Trained model found at {trained_model_path}")

            # Step 2: Copy model to deployment directory
            os.makedirs(self.model_pusher_config.TRAINED_MODEL_DIR, exist_ok=True)
            shutil.copy(trained_model_path, self.model_pusher_config.BEST_MODEL_PATH)

            logging.info(f"Model copied to {self.model_pusher_config.BEST_MODEL_PATH}")

            # Step 3: Create artifact
            model_pusher_artifact = ModelPusherArtifacts(
                model_dir=self.model_pusher_config.TRAINED_MODEL_DIR,
                best_model_path=self.model_pusher_config.BEST_MODEL_PATH
            )

            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
