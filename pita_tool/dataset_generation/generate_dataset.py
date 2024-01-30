"""
Generate dataset with json annotations and images.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from PIL import Image
from pycocotools.coco import COCO

from .annotation import Annotation, AnnotationEncoder
from .dataset import PitaDataset
from .utils.download_dataset import download_annotations
from .utils.watermarks_generator import add_text_watermark, load_fonts


def generate_dataset_anotations(coco_api: COCO, dataset : PitaDataset, image_directory : str) -> None:
    font_names: List[str] = load_fonts()

    with open(dataset.get_metadata_path(), "a") as f:
        for idx, image in enumerate(dataset.image_ids):
            
            image_properties : List = coco_api.loadImgs(image)[0]
            image_path : str = image_directory + "/" + str(image_properties["file_name"])
            
            image_pil : Image = Image.open(image_path)
            
            bbox = [0, 0, 0, 0]

            anotation = Annotation(
                file_name=image_properties["file_name"],
                bbox=bbox,
                id=idx,
                image_id=image_properties["id"],
                category_id=1,
            )

            json.dump(anotation, f, cls=AnnotationEncoder)


# def generate_dataset(
#     dataset_directory: str = "data/train",
#     annotations_file: str = "data/annotations/instances_train2017.json",
#     output_file: str = "metadata.jsonl",
#     size: int = 20_000,
# ) -> None:
#     """
#     Generate a huggingface dataset from a coco annotations file.

#     Args:
#         annotations_file (str): The annotations file to generate the dataset from.
#     """
#     coco_api: COCO = COCO(annotations_file)
#     # download images from coco dataset
#     images: List[int] = coco_api.getImgIds()[:size]
#     download_images(coco_api, dataset_directory, images)

#     # create directory if it doesn't exist recursively
#     if not os.path.exists(dataset_directory):
#         os.makedirs(dataset_directory)

#     # write annotations
#     write_annotations(
#         coco_api=coco_api,
#         dataset_directory=dataset_directory,
#         output_file=output_file,
#         start_index=0,
#         end_index=size,
#     )


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

    with TemporaryDirectory() as temporary_directory:
        annotation_file: str = download_annotations(
            annotions_url=dataset.annotation_url, directory_path=temporary_directory
        )

        coco_api : COCO = COCO(annotation_file)

        dataset.image_ids = coco_api.getImgIds()[: dataset.size]

        # TODO: remove
        image_directory : str = temporary_directory + "/images"
        coco_api.download(image_directory, dataset.image_ids)

        generate_dataset_anotations(coco_api, dataset, image_directory)
