"""
Download the dataset from the given URL and extract the images and annotations.
"""
import os
import sys
import threading
import zipfile
from glob import glob
from io import StringIO
from pathlib import Path
from typing import List

import wget
from pycocotools.coco import COCO


class NoAnnotationsFileFound(Exception):
    """Raised when no annotations are found."""

    def __init__(self, message="No annotations found."):
        self.message = message
        super().__init__(self.message)


class NullIO(StringIO):
    """
    A class that does nothing.
    """

    def write(self, txt):
        pass


def download_annotations(
    annotions_url: str, output_directory: str = "data"
) -> List[str]:
    """Download the annotations from the given URL.

    Args:
        annotions_url (str): The URL to download the annotations from.
    """

    print("Downloading annotations...")
    output_zip = "image_anotations.zip"
    annotations_zip = wget.download(annotions_url, out=output_zip)

    # Extract the annotations
    with zipfile.ZipFile(annotations_zip, "r") as zip_ref:
        zip_ref.extractall(output_directory)

    os.remove(annotations_zip)

    # Get the annotations file
    annotations_json = glob(f"{output_directory}/**/*.json")
    if len(annotations_json) == 0:
        raise NoAnnotationsFileFound("No annotations file found.")

    return Path(annotations_json[0]).absolute()


def download_image(coco_api: COCO, dataset_directory: str, image_id: List[int]) -> None:
    """
    Download one image.

    Args:
        coco_api (COCO): The coco api.
        dataset_directory (str): The dataset directory.
        image_id (int): The image id.
    """
    coco_api.download(dataset_directory, image_id)


def download_images(
    coco_api: COCO, dataset_directory: str, images: List[int], n_jobs: int = -1
) -> None:
    """
    Download multiple images with multiple threads.
    """
    # Check if images are already downloaded
    current_images : List[str] = glob(f"{dataset_directory}/**/*.jpg", recursive=True)
    if len(current_images) == len(images):
        print("Images already downloaded.")
        return

    # Download the images
    print("Downloading images...")
    sys.stdout: NullIO = NullIO()
    n_jobs: int = os.cpu_count() if n_jobs == -1 else n_jobs
    images_per_job: int = len(images) // n_jobs

    threads = []
    for i in range(n_jobs):
        start_index = i * images_per_job
        end_index = (i + 1) * images_per_job

        # Last job
        if i == n_jobs - 1:
            end_index = len(images)

        t = threading.Thread(
            target=download_image,
            args=(coco_api, dataset_directory, images[start_index:end_index]),
        )
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    sys.stdout = sys.__stdout__
