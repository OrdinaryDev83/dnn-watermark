"""
Generate dataset with json annotations and images.
"""

import json
import os
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

from PIL import Image
from pycocotools.coco import COCO

from .annotation import Annotation, AnnotationEncoder
from .dataset import PitaDataset
from .download_dataset import download_annotations, download_images, download_logos
from .utils import DisablePrint
from .watermarks_generator import add_logo_watermark, add_text_watermark, load_fonts

logger = getLogger(__name__)


def generate_text_watermarked_image(
    coco_api: COCO, idx: int, image_id: int, font: str, image_directory: str
) -> None:
    """
    Generate an image with a text watermark.

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

    return watermarked_image, bbox, category, image_properties


def generate_logo_watermarked_image(
    coco_api: COCO,
    idx: int,
    image_id: int,
    image_directory: str,
    logo_path: str,
) -> None:
    """
    Generate an image with a logo watermark.

    Args:
        coco_api (COCO): The coco api.
        idx (int): The image index.
        image_id (int): The image id.
        font (str): The font name.
        image_directory (str): The image directory.
        logo_directory (str): The logo directory.
    """
    image_properties: Dict = coco_api.loadImgs(image_id)[0]
    image_path: str = image_directory + "/" + str(image_properties["file_name"])

    image_pil: Image = Image.open(image_path)
    logo_pil: Image = Image.open(logo_path)
    watermarked_image, bbox, category = add_logo_watermark(image_pil, logo_pil)

    return watermarked_image, bbox, category, image_properties


def generate_labels(
    coco_api: COCO, dataset: PitaDataset, image_directory: str, logo_directory: str
) -> None:
    """
    Generate the dataset annotations.

    Args:
        coco_api (COCO): The coco api.
        dataset (PitaDataset): The PitaDataset object.
        image_directory (str): The image directory.
    """
    font_names: List[str] = load_fonts()
    logo_pathes: List[str] = os.listdir(logo_directory)

    with open(dataset.get_metadata_path(), "a") as f:
        for idx, image_id in enumerate(dataset.image_ids):
            font = font_names[idx % len(font_names)]
            logo_path: str = logo_directory + "/" + logo_pathes[idx % len(logo_pathes)]

            try:
                if idx % 2 == 0:
                    (
                        watermarked_image,
                        bbox,
                        category,
                        image_properties,
                    ) = generate_logo_watermarked_image(
                        coco_api, idx, image_id, image_directory, logo_path
                    )
                else:
                    (
                        watermarked_image,
                        bbox,
                        category,
                        image_properties,
                    ) = generate_text_watermarked_image(
                        coco_api,
                        idx,
                        image_id,
                        font,
                        image_directory,
                    )
            except Exception as e:
                logger.warning(e)
                continue

            if bbox is None:
                logger.warning("Empty bbox in image %s", image_properties["file_name"])
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
            watermarked_image.save(
                dataset.get_image_path(image_properties["file_name"])
            )
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
        download_images(
            coco_api=coco_api,
            output_directory=image_directory,
            images=dataset.image_ids,
        )

        # Download logo images
        logo_directory: str = temporary_directory + "/logos/all/"
        download_logos(directory_path=temporary_directory)

        # remove metadata file if it exists
        if os.path.exists(dataset.get_metadata_path()):
            os.remove(dataset.get_metadata_path())

        generate_labels(coco_api, dataset, image_directory, logo_directory)
