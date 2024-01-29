import os
import random
import string
import cv2
from typing import List, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_fonts() -> List[str]:
    """
    Load all fonts from the system

    Returns:
        List[str]: List of fonts names
    """
    flist = fm.findSystemFonts()

    font_names = [
        fm.FontProperties(fname=fname).get_file().split("\\")[-1]
        for fname in flist
    ]

    return font_names


def load_images():
    path_pictures = os.path.join("data", "pictures")
    data = []

    for k, filename in enumerate(os.listdir(path_pictures)):
        img = Image.open(os.path.join(path_pictures, filename))
        data.append(img.copy())
        img.close()

    return data


def generate_random_string(length: int) -> str:
    """
    Generate a random string of a given length

    Args:
        length (int): The length of the string

    Returns:
        str: The random string
    """

    letters: str = string.ascii_letters + string.digits
    letters += string.whitespace.replace("\n", "").replace("\t", "")
    result_str: str = "".join(random.choice(letters) for _ in range(length))

    return result_str


def _get_position(img_width: int, img_height: int, text_size: int) -> Tuple[int, int]:
    """
    Returns a random position for the watermark

    Args:
        img_width (int): The width of the image
        img_height (int): The height of the image
        text_size (int): The size of the text

    Returns:
        tuple: The position of the watermark
    """
    padding = 0.05
    positions = [
        {"top_left": (img_width * padding, img_height * padding + text_size)},
        {"top_right": (img_width * (1 - padding), img_height * padding + text_size)},
        {"bottom_left": (img_width * padding, img_height * (1 - padding) - text_size)},
        {
            "bottom_right": (
                img_width * (1 - padding),
                img_height * (1 - padding) - text_size,
            )
        },
        {"middle": (img_width // 2, img_height // 2)},  # middle
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
    return 0  # np.random.choice(rotations)

def _get_alpha_from_rotation_and_position(position: dict, rotation: int) -> int:
    """
    Returns a random alpha value for the watermark

    Args:
        position (dict): The position of the watermark
        rotation (int): The rotation angle
    
    Returns:
        int: The alpha value
    """
    possible_alpha_ranges =[
        (255 * 0.3, 255 * 0.6),
        (255 * 0.8, 255),
    ]
    pos_key = list(position.keys())[0]
    if pos_key in ["top_left", "top_right", "bottom_left", "bottom_right"] and rotation in [0, 180]:
        alpha_range = possible_alpha_ranges[1]
    else:
        alpha_range = possible_alpha_ranges[0]

    return np.random.randint(alpha_range[0], alpha_range[1])


def _get_color_from_rotation_and_position(position: dict, rotation: int) -> tuple:
    """
    Returns a random color
    """
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        _get_alpha_from_rotation_and_position(position, rotation),
    )


def get_direction_anchor_from_position(position: dict) -> str:
    """
    Returns the direction and anchor for the text

    Args:
        position (dict): The position of the text

    Returns:
        str: The direction and anchor
    """
    pos_key: str = list(position.keys())[0]

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


def add_text_watermark(img: Image.Image, font_name: str) -> Tuple:
    """
    Add a text watermark to an image

    Args:
        img (Image.Image): The image
        font_name (str): The font name

    Returns:
        tuple: The image with the watermark, the watermark image, the bounding box

    """
    w, h = img.size

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

    dir, anchor = get_direction_anchor_from_position(position)
    draw.text(position_values, txt, fill=color, font=font, direction=dir, anchor=anchor)
    txt_image = txt_image.rotate(rotation)

    draw = ImageDraw.Draw(txt_new_img)
    draw.text(position_values, txt, fill=color, font=font, direction=dir, anchor=anchor)

    txt_new_img = txt_new_img.rotate(rotation)
    combined = Image.alpha_composite(new_img, txt_new_img)

    bbox = txt_new_img.getbbox()
    return combined, bbox, 2


def resize_image_bbox(img, bboxes):
    bboxes = np.array(bboxes)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        bbox[0] = bbox[0] * 224 // img.size[0]
        bbox[1] = bbox[1] * 224 // img.size[1]
        bbox[2] = bbox[2] * 224 // img.size[0]
        bbox[3] = bbox[3] * 224 // img.size[1]
    return img.resize((224, 224)), bboxes


def plot_watermark(img, bbox=None):
    """
    Plot the image with the watermark and the bounding box
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    if bbox is not None:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    ax.add_patch(rect)
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

def remove_background(img: Image.Image):
    img = np.array(img)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: # if logo is already grayscale
        gray = img
    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    # negate mask
    mask = 255 - mask
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return Image.fromarray(result, 'RGBA')

def _get_position_for_logo(img_width: int, img_height: int, logo_height: int, logo_width: int) -> tuple:
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
        {"top_left": (img_width * padding, img_height * padding)},  # top left
        {"top_right": (img_width * (1 - padding) - logo_width, img_height * padding)},  # top right
        {"bottom_left": (img_width * padding, img_height * (1 - padding) - logo_height)},  # bottom left
        {"bottom_right": (img_width * (1 - padding) - logo_width, img_height * (1 - padding) - logo_height)},  # bottom right
        {"middle": (img_width // 2 - logo_width // 2, img_height // 2 - logo_height // 2)}  # middle
    ]
    return np.random.choice(positions)

def add_logo_watermark(img: Image.Image, logo: Image.Image) -> tuple:
    """
    Main to function to add logo watermark to an image
    Args:
        img (Image.Image): image to add watermark to
        logo (Image.Image): logo to add
    Returns:
        tuple: combined image, logo bbox, category
    """
    w, h = img.size
    logo = remove_background(logo)
    max_logo_size = np.random.uniform(0.3, 0.6)
    scale_factor: int = (
        max_logo_size
        / max(logo.width, logo.height)
        * min(img.width, img.height)
    )
    logo_resized = logo.resize(
        (int(logo.width * scale_factor), int(logo.height * scale_factor))
    )
    position: dict = _get_position_for_logo(w, h, logo_resized.height, logo_resized.width)
    rotation: int = _get_rotation_from_position(position)
    position_values = position[list(position.keys())[0]]
    position_values = int(position_values[0]), int(position_values[1])
    alpha = _get_alpha_from_rotation_and_position(position, rotation)
    
    logo_resized = logo_resized.rotate(rotation)
    logo_resized = np.array(logo_resized)
    logo_resized[:, :, 3] *= alpha
    logo_resized = Image.fromarray(logo_resized, "RGBA")

    logo_transformed = Image.new("RGBA", img.size, (0, 0, 0, 0))
    logo_transformed.paste(logo_resized, position_values)
    bbox = logo_resized.getbbox()
    bbox = (*position_values, position_values[0] + bbox[2], position_values[1] + bbox[3])
    
    new_img = img.copy().convert("RGBA")
    combined = Image.alpha_composite(new_img, logo_transformed)
    return combined, bbox, 2

# if __name__ == "__main__":
#     # pos = _get_position(100, 100)
#     # print(pos)
#     # rot = _get_rotation_from_position(pos)
#     # print(rot)
#     # c = _get_color_from_rotation_and_position(pos, rot)
#     # print(c)
#     fonts = load_fonts()
#     img = Image.open("data/pictures/000000000139.jpg")
#     combined, txt_img, bbox = add_text_watermark(img, fonts[0])
#     combined, bbox, _ = add_logo_watermark(img, Image.open("data/logos/amazon.jpg"))
#     plot_watermark(combined, bbox)
