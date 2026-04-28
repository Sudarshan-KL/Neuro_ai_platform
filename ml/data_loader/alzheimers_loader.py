"""
Data loader for Alzheimer’s MRI image dataset (classification).
Assumes folder structure:
  data/alzheimers/<ClassName>/<image files>
"""
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image

def get_alzheimers_image_paths(data_dir: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (image_path, label) tuples for all images in the dataset.
    """
    data_dir = Path(data_dir)
    samples = []
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for img_file in class_dir.glob("*.jpg"):
                samples.append((str(img_file), label))
    return samples

def load_image(img_path: str, size=(224, 224)) -> Image.Image:
    """
    Loads an image and resizes it for model input.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    return img

# Example usage:
if __name__ == "__main__":
    data_dir = "../../data/alzheimers"
    samples = get_alzheimers_image_paths(data_dir)
    print(f"Found {len(samples)} images.")
    img, label = samples[0]
    image = load_image(img)
    image.show()
    print(f"Label: {label}")
