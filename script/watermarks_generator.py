import json
import os
import random
import shutil
import string
from typing import List

import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange


flist = fm.findSystemFonts()
font_names = [
    fm.FontProperties(fname=fname).get_file().split("\\")[-1].lower() for fname in flist
]


def generate_random_string(length: int) -> str:
    letters: str = string.ascii_letters + string.digits
    letters += string.whitespace.replace("\n", "").replace("\t", "")
    result_str: str = "".join(random.choice(letters) for _ in range(length))
    return result_str


def _get_position(img_width, img_height, text_size) -> tuple:
    """
    Returns a position between:
    - top left
    - top right
    - bottom left
    - bottom right
    - middle
    With a padding of 10% of the image size
    """
    padding = 0.05
    positions = [
        {"top_left": (img_width * padding, img_height * padding + text_size)},  # top left
        {"top_right": (img_width * (1 - padding), img_height * padding + text_size)},  # top right
        {"bottom_left": (img_width * padding, img_height * (1 - padding) - text_size)},  # bottom left
        {"bottom_right": (img_width * (1 - padding), img_height * (1 - padding) - text_size)},  # bottom right
        {"middle": (img_width // 2, img_height // 2)}  # middle
    ]
    return np.random.choice(positions)


def _get_rotation_from_position(position: dict) -> int:
    """
    Returns a rotation angle from a position
    """
    rotations = []
    pos_key = list(position.keys())[0]
    if pos_key == "top_left":
        rotations = [0, 45, 90]
    elif pos_key == "top_right":
        rotations = [90, 135, 180]
    elif pos_key == "bottom_left":
        rotations = [0, 270, 315]
    elif pos_key == "bottom_right":
        rotations = [180, 225, 270]
    elif pos_key == "middle":
        rotations = [0]
    return 0 #np.random.choice(rotations)


def _get_color_from_rotation_and_position(position: dict, rotation: int) -> tuple:
    """
    Returns a random color
    """
    possible_alpha_ranges =[
        (255 * 0.25, 255 * 0.6),
        (255 * 0.75, 255),
    ]
    if position in ["top_left", "top_right", "bottom_left", "bottom_right"] and rotation in [0, 180]:
        alpha_range = possible_alpha_ranges[1]
    else:
        alpha_range = possible_alpha_ranges[0]
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(alpha_range[0], alpha_range[1])
    )


def add_text_watermark(img) -> tuple:
    """
    Main function to add a text watermark to an image
    """
    w, h = img.size
    print(w, h)
    txt: str = generate_random_string(np.random.randint(8, 15))
    size: int = np.random.randint(30, min(w, h) // 5)
    position: dict = _get_position(w, h, size)
    rotation: int = _get_rotation_from_position(position)
    color: tuple = _get_color_from_rotation_and_position(position, rotation)
    position_values = list(position.values())[0]

    font = "/system/library/fonts/supplemental/arial.ttf"

    new_img = img.copy().convert("RGBA")
    txt_new_img = Image.new("RGBA", new_img.size, (255, 255, 255, 0))
    txt_image = Image.new("RGBA", img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(txt_image)
    font = ImageFont.truetype(font, size=size)
    print(txt, position, rotation, color)
    draw.text(position_values, txt, fill=color, font=font, direction="ttb")  # direction="ttb" to write vertically, and anchor="ls" or "rs" to write from left or right
    txt_image = txt_image.rotate(rotation)

    draw = ImageDraw.Draw(txt_new_img)
    draw.text(position_values, txt, fill=color, font=font, direction="ttb") # direction="ttb" to write vertically, and anchor="ls" or "rs" to write from left or right
    txt_new_img = txt_new_img.rotate(rotation)
    combined = Image.alpha_composite(new_img, txt_new_img)

    bbox = txt_new_img.getbbox()
    return combined, txt_image, bbox


def plot_watermark(img, txt_img):
    """
    Plot the image with the watermark
    """
    total_txt_image = Image.alpha_composite(img, txt_img)

    fig, ax = plt.subplots(1)
    ax.imshow(total_txt_image)
    plt.show()


if __name__ == "__main__":
    # pos = _get_position(100, 100)
    # print(pos)
    # rot = _get_rotation_from_position(pos)
    # print(rot)
    # c = _get_color_from_rotation_and_position(pos, rot)
    # print(c)
    img = Image.open("../data/pictures/000000000139.jpg")
    combined, txt_img, bbox = add_text_watermark(img)
    plot_watermark(combined, txt_img)

