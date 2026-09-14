from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from occluded_sitting_smplx.types import (
    BackendPrediction,
    BBox,
    CameraParameters,
    SMPLXParameters,
)


def zero_parameters() -> SMPLXParameters:
    return SMPLXParameters(
        global_orient=np.zeros((1, 3), dtype=np.float32),
        body_pose=np.zeros((21, 3), dtype=np.float32),
        left_hand_pose=np.zeros((15, 3), dtype=np.float32),
        right_hand_pose=np.zeros((15, 3), dtype=np.float32),
        jaw_pose=np.zeros((1, 3), dtype=np.float32),
        leye_pose=np.zeros((1, 3), dtype=np.float32),
        reye_pose=np.zeros((1, 3), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        expression=np.zeros((1, 10), dtype=np.float32),
        transl=np.asarray([[0.0, 0.0, 2.5]], dtype=np.float32),
    )


class FakeBackend:
    model_name = "fake-smpler-x"
    model_commit = "test-commit"
    peak_gpu_memory_mb = 128.0

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [BBox(10, 10, 60, 80, 0.9)]
        self.detect_calls = 0

    def detect(self, rgb):
        self.detect_calls += 1
        return self.boxes

    def predict(self, rgb, bbox):
        parameters = zero_parameters()
        return BackendPrediction(
            parameters=parameters,
            camera=CameraParameters((500.0, 500.0), (rgb.shape[1] / 2, rgb.shape[0] / 2)),
            processed_bbox=bbox,
            reference_vertices=self.build_mesh(parameters),
        )

    def build_mesh(self, parameters):
        vertices = np.zeros((10475, 3), dtype=np.float32)
        vertices[:, 0] = np.linspace(-0.5, 0.5, len(vertices))
        vertices[:, 2] = 2.5
        return vertices

    @property
    def faces(self):
        return np.asarray([[0, 1, 2], [2, 3, 0]], dtype=np.int32)


class FakeRenderer:
    def overlay(self, rgb, vertices, faces, camera, output_path):
        Image.fromarray(rgb).save(output_path)

    def preview(self, vertices, faces, output_path, view_size=480):
        Image.new("RGB", (90, 30), "white").save(output_path)


@pytest.fixture
def input_image(tmp_path: Path) -> Path:
    path = tmp_path / "person.png"
    Image.new("RGB", (120, 160), (80, 100, 120)).save(path)
    return path
