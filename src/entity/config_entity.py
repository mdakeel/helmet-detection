import os
from dataclasses import dataclass
from from_root import from_root
from src.constant import *


@dataclass
class DataIngestionConfig:
    def __init__(self):
        # Artifact folder
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)

        # Direct dataset paths (not inside artifacts)
        self.TRAIN_DATA_ARTIFACT_DIR = os.path.join(from_root(), DATA_INGESTION_TRAIN_DIR)
        self.TEST_DATA_ARTIFACT_DIR  = os.path.join(from_root(), DATA_INGESTION_TEST_DIR)
        self.VALID_DATA_ARTIFACT_DIR = os.path.join(from_root(), DATA_INGESTION_VALID_DIR)

        self.UNZIPPED_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR = os.path.join(from_root(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TRAIN_DIR, DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH  = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TEST_DIR, DATA_TRANSFORMATION_TEST_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR  = os.path.join(from_root(), TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
        self.BATCH_SIZE   = TRAINED_BATCH_SIZE
        self.SHUFFLE      = TRAINED_SHUFFLE
        self.NUM_WORKERS  = TRAINED_NUM_WORKERS
        self.EPOCH        = EPOCH
        self.DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.EVALUATED_MODEL_DIR = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.EVALUATED_LOSS_CSV_PATH = os.path.join(self.EVALUATED_MODEL_DIR, MODEL_EVALUATION_FILE_NAME)


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR  = os.path.join(from_root(), TRAINED_MODEL_DIR)
        self.BEST_MODEL_PATH    = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)