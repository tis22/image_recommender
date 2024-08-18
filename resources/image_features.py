import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, ExifTags
import os


def image_rgb_calculation(image):
    """
    Calculates the RGB histogram for an image.

    Args:
        image (np.array): Image array in BGR format.

    Returns:
        np.array: Flattened and normalized RGB histogram.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([rgb_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def image_hsv_calculation(image):
    """
    Calculates the HSV histogram for an image.

    Args:
        image (np.array): Image array in BGR format.

    Returns:
        np.array: Flattened and normalized HSV histogram.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def load_embedding_model():
    """
    Loads the EfficientNet model for embedding calculations.

    This function loads a pre-trained EfficientNet model and prepares it for 
    embedding calculations by removing the final classification layer. It also 
    sets up a preprocessing pipeline to prepare images for input to the model.

    Returns:
        None
    """
    global model, preprocess
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def model_embeddings_calculation(image):
    """
    Calculates the embedding for an image using the pre-trained EfficientNet model.

    Args:
        image (np.array): Image array.

    Returns:
        np.array: Flattened embedding array.
    """
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(input_batch)
    return features.numpy().flatten()


def extract_image_details(image_id, path, resize_size):
    """
    Extracts detailed information from an image including histograms, embeddings, and metadata.

    Args:
        image_id (int): Unique identifier for the image.
        path (str): Path to the image file.
        resize_size (tuple): Desired size for resizing the image.

    Returns:
        dict: Dictionary containing various extracted details about the image.
    """
    try:
        image = cv2.imread(path)
        if image is not None:
            image = cv2.resize(image, resize_size)
            rgb_histogram = image_rgb_calculation(image)
            hsv_histogram = image_hsv_calculation(image)
            model_embedding = model_embeddings_calculation(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            avg_color = np.mean(image_rgb, axis=(0, 1)).tolist()
            avg_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(image_hsv, axis=(0, 1)).tolist()
            file_size = os.path.getsize(path)
            file_type = os.path.splitext(path)[1]

            with Image.open(path) as img:
                resolution = img.size
                dpi = img.info.get("dpi", (0, 0))
                try:
                    exif_data = img._getexif()
                    metadata = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()} if exif_data else {}
                except AttributeError:
                    metadata = {}

            return {
                "ID": image_id,
                "Path": path,
                "RGB_Histogram": rgb_histogram,
                "HSV_Histogram": hsv_histogram,
                "Model_Embedding": model_embedding,
                "Average_Color": avg_color,
                "Brightness": avg_brightness,
                "Average_HSV": avg_hsv,
                "Resolution": resolution,
                "DPI": dpi,
                "File_Size": file_size,
                "File_Type": file_type,
                "Metadata": metadata,
            }
        else:
            print(f"Image at path {path} is None.")
            return None
    except Exception as e:
        print(f"Failed processing {path}: {e}")
        return None
