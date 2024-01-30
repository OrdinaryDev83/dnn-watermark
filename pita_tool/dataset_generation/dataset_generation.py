"""
Generate dataset with json annotations and images.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

from PIL import Image
from pycocotools.coco import COCO

from .annotation import Annotation, AnnotationEncoder
from .dataset import PitaDataset
from .download_dataset import download_annotations, download_images
from .watermarks_generator import add_text_watermark, load_fonts
from .utils import DisablePrint


def generate_text_watermarked_image(
    coco_api: COCO, idx: int, image_id: int, font: str, image_directory: str
) -> None:
    """
    Generate a watermarked text image.

    Args:
        coco_api (COCO): The coco api.
        idx (int): The image index.
        image_id (int): The image id.
        font (str): The font name.
        image_directory (str): The image directory.
    """
    image_properties: Dict = coco_api.loadImgs(image_id)[0]
    image_path: str = image_directory + "/" + str(image_properties["file_name"])

    image_pil: Image = Image.open(image_path)
    watermarked_image, bbox, category = add_text_watermark(image_pil, font)

    return watermarked_image.convert("RGB"), bbox, category, image_properties


def generate_text_labels(
    coco_api: COCO, dataset: PitaDataset, image_directory: str
) -> None:
    """
    Generate the dataset annotations.

    Args:
        coco_api (COCO): The coco api.
        dataset (PitaDataset): The PitaDataset object.
        image_directory (str): The image directory.
    """
    font_names: List[str] = load_fonts()

    with open(dataset.get_metadata_path(), "a") as f:
        for idx, image_id in enumerate(dataset.image_ids):
            font = font_names[idx % len(font_names)]

            # Generate a text watermarked image and check no issues comming from fonts
            try:
                (
                    watermarked_image,
                    bbox,
                    category,
                    image_properties,
                ) = generate_text_watermarked_image(
                    coco_api, idx, image_id, font, image_directory
                )
            except OSError:
                continue

            # Text was not correctly added
            if bbox is None:
                continue

            # Create the annotation
            anotation = Annotation(
                file_name=image_properties["file_name"],
                bbox=bbox,
                id=idx,
                image_id=image_properties["id"],
                category_id=category,
            )

            # Save the image and annotation
            watermarked_image.save(dataset.get_image_path(idx))
            json.dump(anotation, f, cls=AnnotationEncoder)


def generate_dataset(dataset: PitaDataset) -> None:
    """
    Generate a dataset from a PitaDataset object.

    Args:
        dataset (PitaDataset): The PitaDataset object.
    """
    # create the directories if they don't exist with split {directory}/{split}
    dataset_path: Path = dataset.get_path()
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # download the annotations in a temporary directory
    with TemporaryDirectory() as temporary_directory:
        annotation_file: str = download_annotations(
            annotions_url=dataset.annotation_url, directory_path=temporary_directory
        )

        with DisablePrint():
            coco_api: COCO = COCO(annotation_file)

        dataset.image_ids = coco_api.getImgIds()[: dataset.size]

        # Download images in a temporary directory
        image_directory: str = temporary_directory + "/images"
        download_images(coco_api, image_directory, dataset.image_ids)

        generate_text_labels(coco_api, dataset, image_directory)
