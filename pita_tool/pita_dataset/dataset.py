"""
PitaDataset class
"""

import os
from pathlib import Path
from shutil import rmtree
from typing import Tuple
from zipfile import ZipFile


class PitaDataset:
    TRAIN_RANGE: Tuple[int, int] = (0, 73_500)
    VAL_RANGE: Tuple[int, int] = (73_500, 98_000)
    TEST_RANGE: Tuple[int, int] = (98_00, 123_000)

    dataset_directory: str
    split: str
    metadata_file: str
    size: int

    def __init__(
        self,
        dataset_directory: str,
        split: str,
        size: int,
    ) -> None:
        """
        Dataset class that support Object Detection datasets in COCO format.

        Currently only supports outputting a metadata.jsonl file.

        Args:
            dataset_directory (str): The directory of the dataset.
            split (str): The split of the dataset.
            size (int): The size of the dataset.
        """
        self.dataset_directory = dataset_directory

        # check if split is valid
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split {split} not in ['train', 'val', 'test']")

        self.split = split
        self.metadata_file = "metadata.jsonl"
        self.size = size
        self.annotation_url = (
            "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
        )
        self.image_ids = []

    def get_path(self) -> Path:
        """
        Get the path of the dataset.

        Returns:
            Path: The path of the dataset.
        """
        return Path(self.dataset_directory) / self.split

    def get_metadata_path(self) -> Path:
        """
        Get the path of the metadata file.

        Returns:
            Path: The path of the metadata file.
        """
        return Path(self.dataset_directory) / self.split / self.metadata_file

    def get_image_path(self, image_id: int) -> Path:
        """
        Get the path of the image.

        Args:
            image_id (int): The image id.

        Returns:
            Path: The path of the image.
        """
        return Path(self.dataset_directory) / self.split / image_id

    def zip_dataset(self) -> None:
        """
        Zip split inside of dataset directory.
        """
        with ZipFile(str(self.get_path()) + ".zip", "w") as zip:
            for file in os.listdir(self.get_path()):
                zip.write(self.get_path() / file, file)

        # remove the unzipped directory
        rmtree(self.get_path())

    def split_indexes(self, split_name: str) -> Tuple[int, int]:
        """
        Get the indexes of the split.

        Returns:
            Tuple[int, int]: The indexes of the split (start, end).
        """
        # check if split is valid
        if split_name == "train":
            if (self.size > self.TRAIN_RANGE[1] - self.TRAIN_RANGE[0]):
                raise ValueError(f"Size {self.size} is larger than train size {self.TRAIN_RANGE[1]}")
            return self.TRAIN_RANGE[0], self.TRAIN_RANGE[0] + self.size

        if split_name == "val":
            if (self.size > self.VAL_RANGE[1] - self.VAL_RANGE[0]):
                raise ValueError(f"Size {self.size} is larger than val size {self.VAL_RANGE[1]}")
            return self.VAL_RANGE[0], self.VAL_RANGE[0] + self.size

        if split_name == "test":
            if (self.size > self.TEST_RANGE[1] - self.TEST_RANGE[0]):
                raise ValueError(f"Size {self.size} is larger than test size {self.TEST_RANGE[1]}")
            return self.TEST_RANGE[0], self.TEST_RANGE[0] + self.size
