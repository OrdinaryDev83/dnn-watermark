"""
PitaDataset class
"""

from pathlib import Path


class PitaDataset:
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
        if (split not in  ["train", "val", "test"]):
            raise ValueError(f"Split {split} not in ['train', 'val', 'test']")

        self.split = split
        self.metadata_file = "metadata.jsonl"
        self.size = size
        self.annotation_url = "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
        self.image_ids = []

    def get_path(self) -> Path:
        return Path(self.dataset_directory) / self.split

    def get_metadata_path(self) -> Path:
        return Path(self.dataset_directory) / self.split / self.metadata_file

    def zip_dataset(self) -> None:
        """
        Zip the dataset
        # TODO: implement
        """
        pass
