"""
Generate dataset with json annotations and images.
"""

import json
import os
from dataclasses import dataclass
from datetime import date
from json import JSONEncoder
from typing import Dict, List

from pycocotools.coco import COCO
from utils.watermarks_generator import add_text_watermark, load_fonts
from utils.download_dataset import download_images


@dataclass
class Annotation:
    """
    Coco format annotation class.
    """

    file_name: str
    bbox: List[float]
    id: int
    area: float
    image_id: int
    category_id: int


class AnnotationEncoder(JSONEncoder):
    """
    Json encoder for Annotation class.
    """

    def default(self, o):
        return o.__dict__


def generate_annotation(
    file_name: str, bbox: List[float], id: int, image_id: int, category_id: int
) -> Annotation:
    """
    Generate an annotation.

    Args:
        bbox (List[float]): The bounding box of the annotation.
        id (int): The id of the annotation.
        image_id (int): The id of the image.
        category_id (int): The id of the category.
    """
    area = bbox[2] * bbox[3]
    return Annotation(bbox, id, area, image_id, category_id)


def append_information(output_annotations_file: str) -> None:
    """
    Generate the information file.

    Args:
        output_annotations_file (str): The output annotations file.
    """
    if not os.path.exists(os.path.dirname(output_annotations_file)):
        os.makedirs(os.path.dirname(output_annotations_file))

    information: Dict = {
        "info": {
            "year": 2024,
            "version": "1",
            "description": "Watermark Detection Dataset, Image comes from the COCO 2017 dataset.",
            "contributor": "EPITA Students",
            "url": "",
            "date_created": date.today().strftime("%Y/%m/%d"),
        }
    }

    with open(output_annotations_file, "a") as f:
        json.dump(information, f)


def generate_categories(output_annotations_file: str) -> None:
    """
    Generate the categories file for coco annotations file.

    Args:
        output_annotations_file (str): The output annotations file.
    """

    # make directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_annotations_file)):
        os.makedirs(os.path.dirname(output_annotations_file))

    categories: Dict = {
        "categories": [
            {"supercategory": "watermark", "id": 1, "name": "text"},
            {"supercategory": "watermark", "id": 2, "name": "logo"},
        ]
    }

    with open(output_annotations_file, "a") as f:
        json.dump(categories, f)


def write_annotations(
    coco_api: COCO,
    dataset_directory: str,
    output_file: str,
    start_index: int,
    end_index: int,
) -> None:
    """
    Write annotations to a file.

    Args:
        coco_api (COCO): The coco api.
        annotation_index (int): The index of the annotation.
        output_annotations_file (str): The output annotations file.
        start_index (int): The start index of the images.
        end_index (int): The end index of the images.
    """
    images: List[int] = coco_api.getImgIds()[start_index:end_index]
    download_images(coco_api, dataset_directory, images)
    

    font_names: List[str] = load_fonts()

    # with open(output_file, "a") as f:
    #     for idx, image in enumerate(images):
    #         image = coco_api.loadImgs(image)[0]

    #         font_name: str = font_names[idx % len(font_names)]
    #         image_watermarked, bbox, category = add_text_watermark(image, font_name)

    #         anotation = Annotation(
    #             file_name=image["file_name"],
    #             bbox=bbox,
    #             id=0,
    #             area=bbox[2] * bbox[3],
    #             image_id=image["id"],
    #             category_id=category,
    #         )

    #         json.dump(anotation, f, cls=AnnotationEncoder)


def zip_dataset() -> None:
    """
    Zip the dataset.
    """
    pass


def generate_dataset(
    dataset_directory: str = "data/train",
    annotations_file: str = "metadata.jsonl",
    output_file: str = "metadata.jsonl",
    size: int = 20_000,
) -> None:
    """
    Generate a huggingface dataset from a coco annotations file.

    Args:
        annotations_file (str): The annotations file to generate the dataset from.
    """
    coco_api: COCO = COCO(annotations_file)

    # create directory if it doesn't exist recursively
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    # write annotations
    write_annotations(
        coco_api=coco_api,
        dataset_directory=dataset_directory,
        output_file=output_file,
        start_index=0,
        end_index=size,
    )


from utils.download_dataset import download_annotations

# print(download_annotations("http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"))
generate_dataset(annotations_file="data/annotations/image_info_unlabeled2017.json")
# generate_categories("data/annotations/categories.json")j
