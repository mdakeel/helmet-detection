import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import save_object
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts


# Simple dataset class for local images + labels
class HelmetDetection:
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")

        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith((".jpg", ".png"))]
        self.num_classes = 2  # Example: With Helmet, Without Helmet

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + ".txt")

        # Load image
        import cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels (YOLO format: class x_center y_center width height)
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        bboxes.append([x, y, w, h, int(cls)])

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=[b[:4] for b in bboxes], class_labels=[b[4] for b in bboxes])
            image = transformed["image"]
            bboxes = transformed["bboxes"]

        return image, bboxes


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def get_transforms(self, train: bool = False):
        try:
            if train:
                transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE),
                    A.HorizontalFlip(p=HORIZONTAL_FLIP),
                    A.VerticalFlip(p=VERTICAL_FLIP),
                    A.RandomBrightnessContrast(p=RANDOM_BRIGHTNESS_CONTRAST),
                    A.ColorJitter(p=COLOR_JITTER),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            else:
                transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            return transform
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered initiate_data_transformation method")

            # Prepare train dataset
            train_dataset = HelmetDetection(
                root=self.data_ingestion_artifact.train_file_path,
                transforms=self.get_transforms(train=True)
            )
            logging.info("Training dataset prepared")

            # Prepare test dataset
            test_dataset = HelmetDetection(
                root=self.data_ingestion_artifact.test_file_path,
                transforms=self.get_transforms(train=False)
            )
            logging.info("Testing dataset prepared")

            # Save transformed objects
            save_object(self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, train_dataset)
            save_object(self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH, test_dataset)
            logging.info("Train and test transformed objects saved")

            # Artifact
            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                transformed_test_object=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH,
                number_of_classes=train_dataset.num_classes
            )

            logging.info("Exited initiate_data_transformation method")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
