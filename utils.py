import base64
import os
import requests
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_random_exponential

@dataclass
class Bbox:
    x_top: int
    y_top: int 
    x_size: int
    y_size: int

# OpenAI API Key
api_key = os.environ["OPENAI_API_KEY"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# Function to encode the image
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def make_api_request(system_msg: str, base64_image: str, max_tokens: int = 500) -> str:
    """Make an API request to a multimodal model API and return the response."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response_data = response.json()
        
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            raise ValueError("No choices returned in response")
        
        return response_data["choices"][0]["message"]["content"]
    
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")
    except ValueError as e:
        raise RuntimeError(f"Invalid response: {e}")

def calculate_iou(bbox_pred: Bbox, bbox_gt: Bbox) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_left = max(bbox_pred.x_top, bbox_gt.x_top)
    y_top = max(bbox_pred.y_top, bbox_gt.y_top)
    x_right = min(bbox_pred.x_top + bbox_pred.x_size, bbox_gt.x_top + bbox_gt.x_size)
    y_bottom = min(bbox_pred.y_top + bbox_pred.y_size, bbox_gt.y_top + bbox_gt.y_size)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = bbox_pred.x_size * bbox_pred.y_size
    box2_area = bbox_gt.x_size * bbox_gt.y_size

    # Calculate the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou
