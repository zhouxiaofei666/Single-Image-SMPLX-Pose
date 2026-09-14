"""Stable public data structures used by the CLI, Web UI, and backend adapter."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BBox:
    """A bounding box in source-image pixel coordinates (x, y, width, height)."""

    x: float
    y: float
    width: float
    height: float
    score: Optional[float] = None

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "BBox":
        if len(values) not in (4, 5):
            raise ValueError("bbox must contain x,y,width,height[,score]")
        return cls(*[float(value) for value in values])

    def clamp(self, image_width: int, image_height: int) -> "BBox":
        x1 = min(max(self.x, 0.0), float(image_width - 1))
        y1 = min(max(self.y, 0.0), float(image_height - 1))
        x2 = min(max(self.x + self.width, x1), float(image_width))
        y2 = min(max(self.y + self.height, y1), float(image_height))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("bbox is empty after clipping to the image")
        return BBox(x1, y1, x2 - x1, y2 - y1, self.score)

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def as_xywh(self) -> List[float]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class CameraParameters:
    focal: Tuple[float, float]
    principal_point: Tuple[float, float]

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "focal": [float(value) for value in self.focal],
            "principal_point": [float(value) for value in self.principal_point],
        }


@dataclass
class SMPLXParameters:
    """Full, non-PCA SMPL-X parameters in axis-angle representation."""

    global_orient: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    betas: np.ndarray
    expression: np.ndarray
    transl: np.ndarray

    _EXPECTED: Mapping[str, Tuple[int, int]] = field(
        init=False,
        repr=False,
        default_factory=lambda: {
            "global_orient": (1, 3),
            "body_pose": (21, 3),
            "left_hand_pose": (15, 3),
            "right_hand_pose": (15, 3),
            "jaw_pose": (1, 3),
            "leye_pose": (1, 3),
            "reye_pose": (1, 3),
            "betas": (1, 10),
            "expression": (1, 10),
            "transl": (1, 3),
        },
    )

    def __post_init__(self) -> None:
        for name, expected_shape in self._EXPECTED.items():
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(expected_shape)
            if not np.isfinite(value).all():
                raise ValueError(f"SMPL-X parameter {name} contains NaN or infinity")
            setattr(self, name, value)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self._EXPECTED}

    def shape_summary(self) -> Dict[str, List[int]]:
        return {name: list(value.shape) for name, value in self.as_dict().items()}


@dataclass
class BackendPrediction:
    parameters: SMPLXParameters
    camera: CameraParameters
    processed_bbox: BBox
    reference_vertices: Optional[np.ndarray] = None


@dataclass
class ReconstructionResult:
    run_dir: Path
    parameters: SMPLXParameters
    vertices: np.ndarray
    faces: np.ndarray
    bbox: BBox
    camera: CameraParameters
    source_image: Path
    obj_path: Path
    parameters_path: Path
    metadata_path: Path
    overlay_path: Path
    preview_path: Path
    archive_path: Path
    timings: Dict[str, float]
