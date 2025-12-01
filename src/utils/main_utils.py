import os
import sys
import dill
import base64
from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to disk using dill serialization.
    """
    logging.info("Entered save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load a Python object from disk using dill serialization.
    """
    logging.info("Entered load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.
    """
    logging.info("Entered image_to_base64 method of utils")
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        logging.info(f"Image {image_path} converted to base64 string")
        return encoded_string
    except Exception as e:
        raise CustomException(e, sys) from e
