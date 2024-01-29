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


def load_fonts():
    """
    Load all fonts from the system
    """
    flist = fm.findSystemFonts()
    font_names = [
        fm.FontProperties(fname=fname).get_file().split("\\")[-1].lower() for fname in flist
    ]
    print(f"Loaded {len(font_names)} fonts")
    return font_names

def load_images():
    path_pictures = os.path.join("data", "pictures")
    data = []
    for k, filename in enumerate(os.listdir(path_pictures)):
        img = Image.open(os.path.join(path_pictures, filename))
        data.append(img.copy())
        img.close()
    print(f"loaded {len(data)} images")
    return data

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
    TODO: TO FIX
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

def get_direction_anchor_from_position(position: dict) -> str:
    """
    Return direction and anchor from position
    """
    pos_key = list(position.keys())[0]
    if pos_key == "top_left":
        return "ltr", "lt"
    elif pos_key == "top_right":
        return "ltr", "rt"
    elif pos_key == "bottom_left":
        return "ltr", "lb"
    elif pos_key == "bottom_right":
        return "ltr", "rb"
    elif pos_key == "middle":
        return "ltr", "mm"


def add_text_watermark(img, font_name) -> tuple:
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
    position_values = position[list(position.keys())[0]]
    font = ImageFont.truetype(font_name, size=size)

    new_img = img.copy().convert("RGBA")
    txt_new_img = Image.new("RGBA", new_img.size, (255, 255, 255, 0))
    txt_image = Image.new("RGBA", img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(txt_image)
    print(txt, position, rotation, color)
    dir, anchor = get_direction_anchor_from_position(position)
    draw.text(position_values, txt, fill=color, font=font, direction=dir, anchor=anchor)  # direction="ttb" to write vertically, and anchor="ls" or "rs" to write from left or right
    txt_image = txt_image.rotate(rotation)

    draw = ImageDraw.Draw(txt_new_img)
    draw.text(position_values, txt, fill=color, font=font, direction=dir, anchor=anchor) # direction="ttb" to write vertically, and anchor="ls" or "rs" to write from left or right
    txt_new_img = txt_new_img.rotate(rotation)
    combined = Image.alpha_composite(new_img, txt_new_img)

    bbox = txt_new_img.getbbox()
    return combined, txt_image, bbox

def resize_image_bbox(img, bboxes):
    bboxes = np.array(bboxes)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        bbox[0] = bbox[0] * 224 // img.size[0]
        bbox[1] = bbox[1] * 224 // img.size[1]
        bbox[2] = bbox[2] * 224 // img.size[0]
        bbox[3] = bbox[3] * 224 // img.size[1]
    return img.resize((224, 224)), bboxes


def plot_watermark(img, txt_img):
    """
    Plot the image with the watermark
    """
    total_txt_image = Image.alpha_composite(img, txt_img)

    fig, ax = plt.subplots(1)
    ax.imshow(total_txt_image)
    plt.show()

def generate_text_watermarks():
    fonts_names = load_fonts()
    images = load_images()
    print("generating text watermarks")
    X, y = [], []
    for img in tqdm(images):
        font_name = np.random.choice(fonts_names)
        font = ImageFont.truetype(font_name, size=20)
        combined, txt_img, bbox = add_text_watermark(img, font)
        combined, bbox = resize_image_bbox(combined, bbox)
        X.append(np.array(combined))
        y.append(bbox)
    X, y = np.array(X), np.array(y)
    return X, y

if __name__ == "__main__":
    # pos = _get_position(100, 100)
    # print(pos)
    # rot = _get_rotation_from_position(pos)
    # print(rot)
    # c = _get_color_from_rotation_and_position(pos, rot)
    # print(c)
    fonts = load_fonts()
    img = Image.open("data/pictures/000000000139.jpg")
    combined, txt_img, bbox = add_text_watermark(img, fonts[0])
    plot_watermark(combined, txt_img)
