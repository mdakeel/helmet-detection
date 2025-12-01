import os
import sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Directly points to local dataset folders (train/test/valid).
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            train_file_path = self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR
            test_file_path  = self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR
            valid_file_path = self.data_ingestion_config.VALID_DATA_ARTIFACT_DIR

            if not os.path.exists(train_file_path):
                raise FileNotFoundError(f"Train folder not found: {train_file_path}")
            if not os.path.exists(test_file_path):
                raise FileNotFoundError(f"Test folder not found: {test_file_path}")
            if not os.path.exists(valid_file_path):
                raise FileNotFoundError(f"Valid folder not found: {valid_file_path}")

            data_ingestion_artifact = DataIngestionArtifacts(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                valid_file_path=valid_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
