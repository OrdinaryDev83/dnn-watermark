"""
Download the dataset from the given URL and extract the images and annotations.
"""
import os
import tarfile
import threading
import zipfile
from glob import glob
from threading import Thread
from typing import List

import wget
from pycocotools.coco import COCO

from .exceptions import NoAnnotationsFileFound
from .utils import DisablePrint


def download_annotations(annotions_url: str, directory_path: str) -> str:
    """
    Download the annotations from the given URL and extract them.

    Args:
        annotions_url (str): The annotations URL.
        directory_path (str): The directory path.

    Returns:
        str: The path to the annotations file.
    """
    zip_path = directory_path + "/anotations.zip"

    # Download the annotations and extract them
    wget.download(annotions_url, out=zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(directory_path)

    annotations_json: List[str] = glob(f"{directory_path}/**/*.json")

    # Check if there are annotations in the directory
    if len(annotations_json) == 0:
        raise NoAnnotationsFileFound("No annotations file found.")

    return annotations_json[0]


def download_images(
    coco_api: COCO, output_directory: str, images: List[int], verbose: bool = True
) -> None:
    """
    Download multiple images.

    Args:
        coco_api (COCO): The coco api.
        output_directory (str): The output directory.
        images (List[int]): The images ids.
    """

    if verbose:
        coco_api.download(output_directory, images)
    else:
        with DisablePrint():
            coco_api.download(output_directory, images)


def download_logos(directory_path: str):
    """
    Download the logos from the given URL and extract them.

    Logos coming froms:
        - QMUL-OpenLogo: Open Logo Detection Challenge Dataset
            https://hangsu0730.github.io/qmul-openlogo/

    Args:
        output_directory (str): The output directory.
    """

    logo_url: str = "https://docs.google.com/uc?export=download&id=1BXlU4aIu5d9f1hZBosZaWYRNzF72UGaj"

    # Download the annotations and extract them
    tar_path: str = directory_path + "/logo.tar"
    wget.download(url=logo_url, out=tar_path)

    # Extract the tar file
    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(directory_path + "/logos")


# def threaded_download_images(
#     coco_api: COCO, output_directory: str, images: List[int], n_jobs: int = -1
# ) -> None:
#     """
#     Download multiple images using multiple threads.

#     Args:
#         coco_api (COCO): The coco api.
#         output_directory (str): The output directory.
#         images (List[int]): The images ids.
#         n_jobs (int): The number of threads to use.

#     """
#     n_jobs: int = os.cpu_count() if n_jobs == -1 else n_jobs
#     images_per_job: int = len(images) // n_jobs

#     # Multiple threads
#     threads: List[Thread] = []
#     for i in range(n_jobs):
#         start_index: int = i * images_per_job
#         end_index: int = (i + 1) * images_per_job

#         # Last job
#         if i == n_jobs - 1:
#             end_index = len(images)

#         t = threading.Thread(
#             target=download_images,
#             args=(coco_api, output_directory, images[start_index:end_index]),
#         )
#         threads.append(t)
#         t.start()

#     for thread in threads:
#         thread.join()
