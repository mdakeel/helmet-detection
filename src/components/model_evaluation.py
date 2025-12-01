import os
import sys
import torch
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts, ModelEvaluationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
        try:
            # Step 1: Load trained model
            model_path = self.model_trainer_artifact.trained_model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained model not found at {model_path}")

            logging.info(f"Model found at {model_path}")

            # Step 2: Dummy evaluation (replace with actual validation loop)
            # For now, we simulate losses
            losses = {"epoch": [1, 2, 3], "loss": [0.45, 0.32, 0.28]}
            df = pd.DataFrame(losses)

            # Step 3: Save losses to CSV
            os.makedirs(self.model_evaluation_config.EVALUATED_MODEL_DIR, exist_ok=True)
            df.to_csv(self.model_evaluation_config.EVALUATED_LOSS_CSV_PATH, index=False)
            logging.info(f"Losses saved at {self.model_evaluation_config.EVALUATED_LOSS_CSV_PATH}")

            # Step 4: Decide acceptance (simple rule: final loss < 0.3)
            final_loss = df["loss"].iloc[-1]
            is_model_accepted = final_loss < 0.3

            # Step 5: Create artifact
            model_evaluation_artifact = ModelEvaluationArtifacts(
                is_model_accepted=is_model_accepted,
                all_losses=self.model_evaluation_config.EVALUATED_LOSS_CSV_PATH
            )

            logging.info("Exited initiate_model_evaluation method of ModelEvaluation class")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
