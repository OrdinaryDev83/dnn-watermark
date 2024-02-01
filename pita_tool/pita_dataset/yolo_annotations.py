import os
from dataclasses import dataclass
from typing import Dict

import yaml
from PIL import Image

from .annotation import Annotation
from .dataset import PitaDataset


@dataclass
class YOLOConfig:
    path: str
    train: str
    val: str
    test: str
    names: Dict[int, str]
    nc: int


def convert_to_YOLO(
    annotation: Annotation,
    dataset: PitaDataset,
    watermarked_image: Image,
    image_size: int = 512,
) -> None:
    """
    Convert the annotation to YOLO format and size the watermarked image with a YOLO architecture.

    Args:
        annotation (Annotation): The annotation.
        dataset (PitaDataset): The dataset.
        watermarked_image (Image): The watermarked image.
        image_size (int, optional): The image size. Defaults to 512.
    """

    yolo_labels_path: str = f"{dataset.dataset_directory}/{dataset.split}-yolo/labels"
    yolo_images_path: str = f"{dataset.dataset_directory}/{dataset.split}-yolo/images"
    label_annotation_filename: str = f"{annotation.file_name.split('.')[0]}.txt"

    if not os.path.exists(yolo_labels_path):
        os.makedirs(yolo_labels_path)

    if not os.path.exists(yolo_images_path):
        os.makedirs(yolo_images_path)

    with open(f"{yolo_labels_path}/{label_annotation_filename}", "w") as f:
        x = annotation.bbox[0] / image_size
        y = annotation.bbox[1] / image_size
        w = annotation.bbox[2] / image_size
        h = annotation.bbox[3] / image_size
        class_id = annotation.category_id - 1
        f.write(f"{class_id} {x} {y} {w} {h}\n")

    watermarked_image.save(f"{yolo_images_path}/{annotation.file_name}")


def generate_YOLO_config(dataset_directory: str, metadata_yml: str) -> None:
    yolo_pita_config = YOLOConfig(
        path=dataset_directory,
        train="train-yolo/images",
        val="val-yolo/images",
        test="test-yolo/images",
        names={
            0: "logo",
            1: "text",
        },
        nc=2,
    )

    with open(f"{dataset_directory}/{metadata_yml}", "w") as f:
        yaml.dump(yolo_pita_config.__dict__, f, default_flow_style=False)
