"""Image normalization and person-box helpers."""

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import DetectionError, InputImageError
from .types import BBox

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_PIXELS = 50_000_000


def load_normalized_rgb(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InputImageError(
            f"Unsupported image type {path.suffix!r}; use JPG, PNG, WebP, or BMP."
        )
    if not path.is_file():
        raise InputImageError(f"Input image does not exist: {path}")
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            if width < 32 or height < 32:
                raise InputImageError("Input image must be at least 32 x 32 pixels.")
            if width * height > MAX_PIXELS:
                raise InputImageError("Input image is too large (maximum 50 megapixels).")
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise InputImageError(f"Could not decode input image: {path}") from exc


def save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path, format="PNG")


def full_image_bbox(rgb: np.ndarray) -> BBox:
    height, width = rgb.shape[:2]
    return BBox(0.0, 0.0, float(width), float(height))


def choose_largest_bbox(boxes: Sequence[BBox], rgb: np.ndarray) -> BBox:
    if not boxes:
        raise DetectionError(
            "No person was detected. Crop the image to one person and use full-image mode."
        )
    height, width = rgb.shape[:2]
    return max(boxes, key=lambda item: item.area).clamp(width, height)
