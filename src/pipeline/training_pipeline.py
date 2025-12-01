import sys
from src.logger import logging
from src.exception import CustomException

# Import configs
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

# Import artifacts
from src.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts
)

# Import components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def run_pipeline(self):
        logging.info("===== Training Pipeline Started =====")
        try:
            # Step 1: Data Ingestion
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact: DataIngestionArtifacts = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            data_transformation = DataTransformation(self.data_transformation_config, data_ingestion_artifact)
            data_transformation_artifact: DataTransformationArtifacts = data_transformation.initiate_data_transformation()

            # Step 3: Model Training
            model_trainer = ModelTrainer(self.model_trainer_config, data_transformation_artifact)
            model_trainer_artifact: ModelTrainerArtifacts = model_trainer.initiate_model_trainer()

            # Step 4: Model Evaluation
            model_evaluation = ModelEvaluation(self.model_evaluation_config, model_trainer_artifact)
            model_evaluation_artifact: ModelEvaluationArtifacts = model_evaluation.initiate_model_evaluation()

            # Step 5: Model Pusher (only if accepted)
            if model_evaluation_artifact.is_model_accepted:
                model_pusher = ModelPusher(self.model_pusher_config, model_trainer_artifact)
                model_pusher_artifact: ModelPusherArtifacts = model_pusher.initiate_model_pusher()
                logging.info(f"Model pushed successfully: {model_pusher_artifact}")
            else:
                logging.info("Model not accepted. Skipping push step.")

            logging.info("===== Training Pipeline Completed Successfully =====")

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
