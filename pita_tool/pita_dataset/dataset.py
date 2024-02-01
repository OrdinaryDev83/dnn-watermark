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
    VAL_RANGE: Tuple[int, int] = (73_501, 98_000)
    TEST_RANGE: Tuple[int, int] = (98_001, 123_000)

    dataset_directory: str
    split: str
    metadata_file: str
    size: int
    dataset_format: str

    def __init__(
        self,
        dataset_directory: str,
        split: str,
        size: int,
        dataset_format: str = "coco",
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

        if dataset_format not in ["coco", "yolo"]:
            raise ValueError(f"Format {dataset_format} not in ['coco', 'yolo']")

        self.dataset_format = dataset_format

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

    def zipdir(self, path: str, ziph: ZipFile) -> None:
        """
        Zip a directory.

        Args:
            path (str): The path of the directory.
            ziph (ZipFile): The ZipFile object.
        """

        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    arcname=os.path.relpath(
                        os.path.join(root, file), os.path.join(path, "..")
                    ),
                )

    def zip_dataset(self) -> None:
        """
        Zip split inside of dataset directory.
        """

        if self.dataset_format == "coco":
            zip_path: str = str(self.get_path()) + ".zip"
            dataset_path: Path = self.get_path()

            with ZipFile(zip_path, "w") as zip:
                for content in os.listdir(dataset_path):
                    zip.write(dataset_path / content, content)

        elif self.dataset_format == "yolo":
            zip_path: str = str(self.get_path()) + "-yolo.zip"
            dataset_path: Path = Path(str(self.get_path()) + "-yolo")
            print(dataset_path)

            with ZipFile(zip_path, "w") as zip:
                zip.mkdir(f"{self.split}-yolo", mode=0o755)
                self.zipdir(dataset_path, zip)

        # remove the unzipped directory
        if self.dataset_format == "yolo":
            rmtree(dataset_path)

        rmtree(self.get_path())

    def split_indexes(self, split_name: str) -> Tuple[int, int]:
        """
        Get the indexes of the split.

        Returns:
            Tuple[int, int]: The indexes of the split (start, end).
        """
        # check if split is valid
        if split_name == "train":
            if self.size > self.TRAIN_RANGE[1] - self.TRAIN_RANGE[0]:
                raise ValueError(
                    f"Size {self.size} is larger than train size {self.TRAIN_RANGE[1]}"
                )
            return self.TRAIN_RANGE[0], self.TRAIN_RANGE[0] + self.size

        if split_name == "val":
            if self.size > self.VAL_RANGE[1] - self.VAL_RANGE[0]:
                raise ValueError(
                    f"Size {self.size} is larger than val size {self.VAL_RANGE[1]}"
                )
            return self.VAL_RANGE[0], self.VAL_RANGE[0] + self.size

        if split_name == "test":
            if self.size > self.TEST_RANGE[1] - self.TEST_RANGE[0]:
                raise ValueError(
                    f"Size {self.size} is larger than test size {self.TEST_RANGE[1]}"
                )
            return self.TEST_RANGE[0], self.TEST_RANGE[0] + self.size
