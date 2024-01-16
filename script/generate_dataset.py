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
names = [
    fm.FontProperties(fname=fname).get_file().split("\\")[-1].lower() for fname in flist
]
for name in names[::]:
    try:
        font = ImageFont.truetype(name)
    except Exception:
        names.remove(name)
print(f"Loaded {len(names)} fonts")

path_pictures = os.path.join("data", "pictures")

data = []
for k, filename in enumerate(os.listdir(path_pictures)):
    img = Image.open(os.path.join(path_pictures, filename))
    data.append(img.copy())
    img.close()

def add_text_watermark(img, txt: str, position: tuple, color: tuple, fontname: str, orientation: int, size: int) -> tuple:
    new_img = img.copy().convert("RGBA")
    txt_new_img = Image.new("RGBA", new_img.size, (255, 255, 255, 0))
    txt_image = Image.new("RGBA", img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(txt_image)
    font = ImageFont.truetype(fontname, size)
    draw.text(position, txt, fill=(0, 0, 0, 255), font=font, anchor="ms")
    txt_image = txt_image.rotate(orientation)

    draw = ImageDraw.Draw(txt_new_img)
    draw.text(position, txt, fill=color, font=font, anchor="ms")
    txt_new_img = txt_new_img.rotate(orientation)
    combined = Image.alpha_composite(new_img, txt_new_img)

    bbox = txt_new_img.getbbox()

    return combined, txt_image, bbox


def generate_random_string(length: int) -> str:
    letters: str = string.printable
    letters += string.whitespace.replace("\n", "").replace("\t", "")
    result_str: str = "".join(random.choice(letters) for i in range(length))
    return result_str


def add_random_text_watermark(img: Image.Image) -> Image.Image:
    w, h = img.size
    txt: str = generate_random_string(np.random.randint(1, 25))
    position: tuple[int, int] = (np.random.randint(0, w), np.random.randint(0, h))
    color: tuple[int, int, int, int] = (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(15, 255),
    )
    fontname: str = np.random.choice(names)
    orientation: int = np.random.choice([0, 90, 180, 270])
    size: int = np.random.randint(5, min(w, h) // 3)
    try:
        # it may fail if some characters from txt are not supported by the font
        return add_text_watermark(img, txt, position, color, fontname, orientation, size)
    except Exception:
        return img, None, None


def add_random_text_watermarks(img: Image.Image, k: int) -> tuple:
    total_txt_image: Image.Image = Image.new("RGBA", img.size, (255, 255, 255, 0))
    watermark_image: Image.Image = img.copy().convert("RGBA")

    bboxes: list = []
    labels: list = []
    while k > 0:
        watermark_image, txt_image, bbox = add_random_text_watermark(watermark_image)
        if bbox is None:
            continue
        total_txt_image = Image.alpha_composite(total_txt_image, txt_image)
        bboxes.append(bbox)
        labels.append("text")
        k -= 1
    return watermark_image, total_txt_image, bboxes, labels


X_data_watermarked = []
y_data_watermarked = []
y_data_bbox = []
y_data_labels = []

txt_per_image = 1

print("Applying Text Watermarks...")
for img in tqdm(data):
    watermark_image, total_txt_image, bboxes, labels = add_random_text_watermarks(
        img, txt_per_image
    )

    bboxes = np.array(bboxes)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        bbox[0] = bbox[0] * 224 // img.size[0]
        bbox[1] = bbox[1] * 224 // img.size[1]
        bbox[2] = bbox[2] * 224 // img.size[0]
        bbox[3] = bbox[3] * 224 // img.size[1]
    watermark_image = watermark_image.resize((224, 224))
    total_txt_image = total_txt_image.resize((224, 224))

    watermark_image = watermark_image.convert("RGB")

    total_txt_image = np.average(np.array(total_txt_image), axis=2)

    X_data_watermarked.append(np.array(watermark_image))
    y_data_watermarked.append(np.array(total_txt_image))
    y_data_bbox.append(bboxes)
    y_data_labels.append(labels)


X_data_watermarked = np.array(X_data_watermarked)
y_data_watermarked = np.array(y_data_watermarked)


X_train, X_val, y_train, y_val = train_test_split(
    X_data_watermarked, y_data_bbox, test_size=0.2, random_state=42, shuffle=False
)


def save_data(
    X_data: np.ndarray,
    y_data: List[List[str]],
    y_data_bbox: List[List[List[float]]],
    filename: str,
    label: str,
    split: str,
    zip: bool = False,
) -> None:
    print(f"Saving {split} data...")
    origin_path = os.path.join("data", "dataset", split)

    if os.path.exists(origin_path):
        shutil.rmtree(origin_path)
    os.makedirs(origin_path)
    #os.makedirs(os.path.join(origin_path, "images"))
    #os.makedirs(os.path.join(origin_path, "labels"))
    jsonl = ""
    for i in trange(len(X_data)):
        img = Image.fromarray(X_data[i])
        img_filename = filename + "_" + split + "_" + str(i) + ".jpg"
        img_path = os.path.join(origin_path, img_filename)
        img.save(img_path)
        bbox_list = []
        for j in range(len(y_data_bbox[i])):
            bbox = y_data_bbox[i][j]
            bbox_list.append([
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2]),
                            int(bbox[3]),
                        ])
        json_dict = {"file_name": img_filename, "label": label, "bbox": bbox_list}
        jsonl += json.dumps(json_dict) + "\n"

    with open(os.path.join(origin_path, "metadata.jsonl"), "w") as f:
        f.write(jsonl)
    if zip:
        shutil.make_archive(origin_path, "zip", origin_path)


save_data(X_train, y_train, y_data_bbox, "watermarked", "text", "train", zip=True)
save_data(X_val, y_val, y_data_bbox, "watermarked", "text", "val", zip=True)

logos = []
for filename in os.listdir(os.path.join("data", "logos")):
    img = Image.open(os.path.join("data", "logos", filename))
    logos.append(img.copy())
    img.close()


def remove_background(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    return img


print("Removing backgrounds from logos...")
no_bg_logos = [remove_background(img) for img in tqdm(logos)]


def add_logo_watermark(
    img: Image.Image,
    logo_img: Image.Image,
    position: tuple,
    orientation: float,
    size: float,
    opacity: int,
) -> tuple:
    new_img = img.copy().convert("RGBA")

    logo_resized = logo_img.convert("RGBA")
    logo_resized = logo_img.resize(
        (int(logo_img.width * size), int(logo_img.height * size))
    )

    alpha = opacity / 255
    logo_resized = np.array(logo_resized)
    logo_resized[:, :, 3] = logo_resized[:, :, 3] * alpha
    logo_resized = Image.fromarray(logo_resized)

    logo_transformed = Image.new("RGBA", img.size, (0, 0, 0, 0))
    logo_resized = logo_resized.rotate(orientation, expand=1)
    logo_transformed.paste(logo_resized, position)

    bbox = logo_resized.getbbox()

    combined = Image.alpha_composite(new_img, logo_transformed)

    return combined, logo_transformed, (*position, bbox[2], bbox[3])


def add_random_logo_watermark(img: Image.Image) -> Image.Image:
    w, h = img.size

    logo_img = no_bg_logos[np.random.randint(0, len(no_bg_logos))]

    max_logo_size = np.random.uniform(0.1, 1)
    scale_factor = (
        max_logo_size
        / max(logo_img.width, logo_img.height)
        * min(img.width, img.height)
    )
    new_size = (int(logo_img.width * scale_factor), int(logo_img.height * scale_factor))

    max_x = w - new_size[0]
    max_y = h - new_size[1]
    position = (np.random.randint(0, max_x), np.random.randint(0, max_y))

    orientation = np.random.choice([0, 90, 180, 270])
    opacity = np.random.randint(15, 255)

    return add_logo_watermark(
        img, logo_img, position, orientation, scale_factor, opacity
    )


def add_random_logo_watermarks(img: Image.Image, k: int):
    total_logo_image: Image.Image = Image.new("RGBA", img.size, (255, 255, 255, 0))
    watermark_image: Image.Image = img.copy().convert("RGBA")

    bboxes: List[List[int]] = []
    labels: List[str] = []
    for _ in range(k):
        watermark_image, logo_image, bbox = add_random_logo_watermark(watermark_image)
        total_logo_image = Image.alpha_composite(total_logo_image, logo_image)
        if bbox:
            bboxes.append(bbox)
            labels.append("logo")

    return watermark_image, total_logo_image, bboxes, labels


X_logos_watermarked = []
y_logos_watermarked = []
y_logos_bbox = []
y_logo_data_labels = []

print("Applying Logo Watermarks...")
for img in tqdm(data):
    watermark_image, total_logo_image, bboxes, labels = add_random_logo_watermarks(
        img, 1
    )

    bboxes = np.array(bboxes)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        bbox[0] = bbox[0] * 224 // img.size[0]
        bbox[1] = bbox[1] * 224 // img.size[1]
        bbox[2] = bbox[2] * 224 // img.size[0]
        bbox[3] = bbox[3] * 224 // img.size[1]
    watermark_image = watermark_image.resize((224, 224))
    total_logo_image = total_logo_image.resize((224, 224))

    watermark_image = watermark_image.convert("RGB")
    y_logo_data_labels.append(labels)

    total_logo_image = np.average(np.array(total_logo_image), axis=2)

    X_logos_watermarked.append(np.array(watermark_image))
    y_logos_watermarked.append(np.array(total_logo_image))
    y_logos_bbox.append(bboxes)

X_logos_train, X_logos_val, y_logos_train, y_logos_val = train_test_split(
    X_logos_watermarked, y_logos_bbox, test_size=0.2, random_state=42, shuffle=False
)

save_data(
    X_logos_train, y_logos_train, y_logos_bbox, "watermarked", "logo", "train", zip=True
)
save_data(
    X_logos_val, y_logos_val, y_logos_bbox, "watermarked", "logo", "val", zip=True
)
