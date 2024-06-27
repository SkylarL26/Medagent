from utils import encode_image, make_api_request, Bbox, calculate_iou
import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional
import re
import numpy as np

system_msg = """
You are a professional radiologist and pulmonologist capable of analyzing chest x-ray 
images and identifying what the most pressing issue is. You will be given a 1024 (max_y) 
x 1024 (max_x) chest x-ray as input. 

You are to use the following procedure on each of the chest x-rays: 
    1) Search for the following abnormalities in the chest x-ray: atelectasis, effusion, 
    cardiomegaly, infiltrate, mass, nodule, pneumonia, pneumothorax.
    2) Find the coordinates and size of the area that is most affected by that 
    abnormality. If there are multiple problematic areas present, select the one that is most prominent.

Return: coordinates <x_top, y_top> and size: <x_size, y_size>. 

Include an explanation of what you see, and then you must include your final diagnosis
in the format: 

<diagnosis>
ABNORMALITY: {True/False} (true if abnormal, false if not abnormal)
COORDINATES: (x_top, y_top)
SIZE: (x_size, y_size)
</diagnosis>
"""


def extract_info(diagnosis: str) -> tuple[bool, Optional[Bbox]]:
    """
    Extracts information from a given text string.

    :param text: Text string containing the diagnosis information
    :return: A dictionary with the extracted information
    """
    # Extract ABNORMALITY
    abnormality_match = re.search(r"ABNORMALITY:\s*(\w+)", diagnosis)
    if not abnormality_match:
        return False, None
    abnormality = abnormality_match.group(1) == "True"

    coordinates_match = re.search(r"COORDINATES:\s*\((\d+),\s*(\d+)\)", diagnosis)
    if not coordinates_match:
        return False, None
    x_top, y_top = (
        int(coordinates_match.group(1)),
        int(coordinates_match.group(2)),
    )

    # Extract SIZE
    size_match = re.search(r"SIZE:\s*\((\d+),\s*(\d+)\)", diagnosis)
    if not size_match:
        return False, None

    x_size, y_size = (
        int(size_match.group(1)),
        int(size_match.group(2)),
    )

    return abnormality, Bbox(x_top=x_top, y_top=y_top, x_size=x_size, y_size=y_size)


def eval_image(image_path: Path, bbox_gt: Bbox) -> tuple[bool, Optional[float]]:
    """
    Given an image path and a ground truth bounding box, detect the abnormality
    and return the predicted abnormality.
    """
    base64_image = encode_image(image_path)
    diagnosis = make_api_request(system_msg, base64_image)
    abnormality, bbox_pred = extract_info(diagnosis)
    if not abnormality:
        return abnormality, None
    print(diagnosis)
    # Render the bounding boxes on the image
    render_bbox(image_path, bbox_pred, bbox_gt)
    return abnormality, calculate_iou(bbox_pred, bbox_gt)


def render_bbox(
    image_path: Path,
    bbox_pred: Bbox,
    bbox_gt: Bbox,
    img_path: Path = Path("./predictions"),
):
    """
    Given an image path and bounding boxes, render the bounding boxes on the image.
    """
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # Open the image file
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw the predicted bounding box in red
        draw.rectangle(
            [
                (bbox_pred.x_top, bbox_pred.y_top),
                (
                    bbox_pred.x_top + bbox_pred.x_size,
                    bbox_pred.y_top + bbox_pred.y_size,
                ),
            ],
            outline="red",
            width=2,
        )
        draw.text((bbox_pred.x_top, bbox_pred.y_top - 10), "predicted", fill="red")

        # Draw the ground truth bounding box in green
        draw.rectangle(
            [
                (bbox_gt.x_top, bbox_gt.y_top),
                (bbox_gt.x_top + bbox_gt.x_size, bbox_gt.y_top + bbox_gt.y_size),
            ],
            outline="green",
            width=2,
        )
        draw.text((bbox_gt.x_top, bbox_gt.y_top - 10), "ground truth", fill="green")

        img.save(img_path / f"{image_path.stem}_rendered.png")


def main():
    labels = pd.read_csv("BBox_List_2017.csv")
    image_dir = Path("images")
    images = os.listdir(image_dir)
    ious = []
    for _, row in labels.iterrows():
        row_list = row.to_list()
        image_path = row_list[0]
        if image_path not in images:
            print(f"Missing image: {image_path}. Skipping...")
            continue
        diagnosis = row_list[1]
        bbox_gt = Bbox(
            x_top=int(row_list[2]),
            y_top=int(row_list[3]),
            x_size=int(row_list[4]),
            y_size=int(row_list[5]),
        )
        try:
            abnormality, iou = eval_image(image_dir / image_path, bbox_gt)
            if iou is not None: 
                ious.append(iou)
            print(f"Image: {image_path}, Abnormality: {abnormality}, IoU: {iou}")
        except Exception as e:
            print(f"Failed to evaluate image: {image_path}. Error: {e}. Continuing...")
            continue
    
    mean = np.mean(ious)
    std = np.std(ious)
    print(ious)
    print(len(ious))
    print(f"Mean: {mean}, Standard deviation: {std}")

# Make sure to define the eval_image function and Bbox class, and import necessary modules
# before running the main function.
if __name__ == "__main__":
    main()
