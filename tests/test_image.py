from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from occluded_sitting_smplx.errors import DetectionError, InputImageError
from occluded_sitting_smplx.image import choose_largest_bbox, load_normalized_rgb
from occluded_sitting_smplx.types import BBox


def test_exif_orientation_is_applied(tmp_path: Path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 80), "red").save(path, exif=exif)
    rgb = load_normalized_rgb(path)
    assert rgb.shape == (40, 80, 3)


def test_unsupported_extension_is_rejected(tmp_path: Path):
    path = tmp_path / "person.gif"
    path.write_bytes(b"GIF89a")
    with pytest.raises(InputImageError, match="Unsupported"):
        load_normalized_rgb(path)


def test_choose_largest_bbox_clamps_to_image():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    selected = choose_largest_bbox([BBox(-5, -5, 80, 80), BBox(20, 20, 10, 10)], rgb)
    assert selected.as_xywh() == [0.0, 0.0, 75.0, 75.0]


def test_no_person_has_actionable_error():
    with pytest.raises(DetectionError, match="full-image mode"):
        choose_largest_bbox([], np.zeros((64, 64, 3), dtype=np.uint8))
