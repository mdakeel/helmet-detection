import os
import torch
from datetime import datetime

# Timestamp for unique artifact folders
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Root artifacts directory
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)

# -----------------------------
# Data Ingestion Constants
# -----------------------------
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"

# Local dataset splits
DATA_INGESTION_TRAIN_DIR = os.path.join("dataset", "train")
DATA_INGESTION_TEST_DIR  = os.path.join("dataset", "test")
DATA_INGESTION_VALID_DIR = os.path.join("dataset", "valid")

# Project name (instead of helmet)
RAW_FILE_NAME = "src"

# -----------------------------
# Data Transformation Constants
# -----------------------------
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
DATA_TRANSFORMATION_TRAIN_DIR = "Train"
DATA_TRANSFORMATION_TEST_DIR  = "Test"

DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.pkl"
DATA_TRANSFORMATION_TEST_FILE_NAME  = "test.pkl"

DATA_TRANSFORMATION_TRAIN_SPLIT = "train"
DATA_TRANSFORMATION_TEST_SPLIT  = "test"

# Image preprocessing parameters
INPUT_SIZE = 416
HORIZONTAL_FLIP = 0.3
VERTICAL_FLIP = 0.3
RANDOM_BRIGHTNESS_CONTRAST = 0.1
COLOR_JITTER = 0.1
BBOX_FORMAT = "yolo"   # since we are using YOLO txt labels

# -----------------------------
# Model Training Constants
# -----------------------------
TRAINED_MODEL_DIR  = "TrainedModel"
TRAINED_MODEL_NAME = "best.pt"

TRAINED_BATCH_SIZE = 2
TRAINED_SHUFFLE    = False
TRAINED_NUM_WORKERS = 1
EPOCH = 1

# -----------------------------
# Model Evaluation Constants
# -----------------------------
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
MODEL_EVALUATION_FILE_NAME     = "loss.csv"

# -----------------------------
# Common Constants
# -----------------------------
use_cuda = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if use_cuda else "cpu")

APP_HOST = "0.0.0.0"
APP_PORT = 8080

# -----------------------------
# Prediction Constants
# -----------------------------
PREDICTION_CLASSES = ["With Helmet", "Without Helmet"]
