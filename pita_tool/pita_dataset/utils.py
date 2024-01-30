"""
This file contains utility functions for the dataset generation.
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image


def resize_image(image: Image, width: int, height: int) -> Image:
    """
    Resize an image to a specific size.

    Args:
        image (Image): The image to resize.
        size (Tuple[int, int]): The size to resize the image to.

    Returns:
        Image: The resized image.
    """
    image = cv2.resize(np.array(image), (width, height))
    return Image.fromarray(image)


class DisablePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
