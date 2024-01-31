from typing import List, Tuple
import PIL
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def get_models_available() -> List[str]:
    """
    Return the list of available models

    Returns:
        List[str]: List of available models
    """
    return ["hustvl/yolos-tiny"]

def load_model(model_name: str) -> Tuple[AutoImageProcessor, AutoModelForObjectDetection]:
    """
    Load the model with the given name

    Args:
        model_name (str): Model name to load
    """
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)

    return image_processor, model

def detect_object(image_processor: AutoImageProcessor, model: AutoModelForObjectDetection, img: PIL.Image) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """
    Take image as input and return the list of bounding boxes and their labels

    Args:
        model (Torch): Model to use for detection
        img (PIL.Image): Input image
    
    Returns:
        Tuple[PIL.Image, List[Tuple[Tuple[int, int, int, int], str]]]: Output image with bounding boxes and labels
    """
    inputs = image_processor(img, return_tensors="pt")
    outputs = model(**inputs)
    target_size = (img.shape[0], img.shape[1])
    target_sizes = torch.tensor([img.shape[0], img.shape[1]]).expand(1, 2)
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    res = []
    for box, label in zip(results["boxes"], results["labels"]):
        boxlist = box.tolist()
        res.append(((round(boxlist[0]), round(boxlist[1]), round(boxlist[2]), round(boxlist[3])), model.config.id2label[label.item()]))
    return res

def detect_watermark(model_name: str, img: PIL.Image) -> Tuple[PIL.Image, List[Tuple[Tuple[int, int, int, int], str]]]:
    """
    Take image as input and return the list of bounding boxes and their labels

    Args:
        model_name (str): Model name to use for detection
        img (PIL.Image): Input image
    
    Returns:
        Tuple[PIL.Image, List[Tuple[Tuple[int, int, int, int], str]]]: Output image with bounding boxes and labels
    """
    image_processor, model = load_model(model_name)
    boxes = detect_object(image_processor, model, img)
    return img, boxes