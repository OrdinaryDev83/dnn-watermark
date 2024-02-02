"""
Preprocess the CLWD dataset to be used in the training of the model
"""

import json
import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_area, box_convert, masks_to_boxes
from torchvision.transforms import functional as F
from tqdm import tqdm


class CLWD(Dataset):
    """
    Custom dataset class for the CLWD dataset
    """

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.masks_dir = mask_dir
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, f"{index}.jpg")
        image = read_image(image_path)

        mask_path = os.path.join(self.masks_dir, f"{index}.png")
        mask = read_image(mask_path)
        try:
            bbox = masks_to_boxes(mask)
        except RuntimeError:
            bbox = None

        target = 0  # logo class

        return image, bbox, target

    def __len__(self):
        return len(os.listdir(self.img_dir))


def dataset_to_YOLO(dataset: CLWD) -> None:
    """
    Convert the dataset to YOLO format and save it in the data/train-yolo directory

    Args:
        dataset : CLWD : The dataset to convert
    """

    # create the directory
    for dir in ["CLWD_dataset/train-yolo/images", "CLWD_dataset/train-yolo/labels"]:
        os.makedirs(dir, exist_ok=True)

    # convert the dataset to YOLO format
    for i in tqdm(range(1, len(dataset) + 1)):
        image, bbox, target = dataset[i]
        if bbox is None:
            continue
        yolo_bbox = box_convert(bbox, in_fmt="xyxy", out_fmt="cxcywh")
        yolo_bbox = yolo_bbox / image.size(1)

        yolo_bbox = yolo_bbox.unique(dim=0)

        image = F.to_pil_image(image)
        image.save(f"CLWD_dataset/train-yolo/images/{i}.jpg")

        with open(f"CLWD_dataset/train-yolo/labels/{i}.txt", "w") as f:
            f.write(f"{target} {' '.join(map(str, yolo_bbox[0].tolist()))}\n")


def dataset_to_hf(dataset: CLWD) -> None:
    """
    Convert the dataset to YOLO format and save it in the data/train-hf directory

    Args:
        dataset : CLWD : The dataset to convert
    """

    # create the directory
    os.makedirs("CLWD_dataset/train/images", exist_ok=True)

    for i in tqdm(range(1, len(dataset) + 1)):
        image, bbox, target = dataset[i]
        if bbox is None:
            continue
        # save the image
        image = F.to_pil_image(image)
        image.save(f"CLWD_dataset/train/images/{i}.jpg")
        area = box_area(bbox).tolist()[0]
        bbox = bbox.unique(dim=0)[0]

        # write to metadata.json file
        metadata = {
            "file_name": f"{i}.jpg",
            "bbox": bbox.tolist(),
            "id": i,
            "area": area,
            "category_id": target,
        }

        with open("CLWD_dataset/train/metadata.json", "a") as f:
            f.write(json.dumps(metadata))
            f.write("\n")


if __name__ == "__main__":
    dataset = CLWD("/Users/quentinfisch/Downloads/CLWD/train/images", "/Users/quentinfisch/Downloads/CLWD/train/masks")
    dataset_to_YOLO(dataset)
