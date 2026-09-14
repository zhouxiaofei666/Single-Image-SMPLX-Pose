import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeBackend, FakeRenderer

from occluded_sitting_smplx.errors import DetectionError, ReconstructionError
from occluded_sitting_smplx.reconstructor import Reconstructor


def test_complete_auto_pipeline(input_image: Path, tmp_path: Path):
    backend = FakeBackend()
    reconstructor = Reconstructor(
        backend=backend, renderer=FakeRenderer(), output_root=tmp_path / "outputs"
    )
    result = reconstructor.reconstruct(input_image, run_id="auto-test")

    assert backend.detect_calls == 1
    assert result.vertices.shape == (10475, 3)
    assert result.obj_path.is_file()
    assert result.parameters_path.is_file()
    assert result.overlay_path.is_file()
    assert result.preview_path.is_file()
    assert result.archive_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["person_selection"]["mode"] == "auto_largest_person"
    assert metadata["mesh"]["vertex_count"] == 10475
    assert metadata["mesh"]["regeneration_max_abs_error"] == 0.0
    assert metadata["runtime"]["peak_gpu_memory_mb"] == 128.0
    assert metadata["model"]["detail_policy"]["face_expression"] == "neutralized"
    with zipfile.ZipFile(result.archive_path) as archive:
        assert "mesh.obj" in archive.namelist()


def test_full_image_bypasses_detector(input_image: Path, tmp_path: Path):
    backend = FakeBackend(boxes=[])
    result = Reconstructor(
        backend=backend, renderer=FakeRenderer(), output_root=tmp_path / "outputs"
    ).reconstruct(input_image, full_image=True, run_id="full-image")
    assert backend.detect_calls == 0
    assert result.bbox.as_xywh() == [0.0, 0.0, 120.0, 160.0]


def test_no_detection_preserves_normalized_input(input_image: Path, tmp_path: Path):
    output_root = tmp_path / "outputs"
    with pytest.raises(DetectionError):
        Reconstructor(
            backend=FakeBackend(boxes=[]),
            renderer=FakeRenderer(),
            output_root=output_root,
        ).reconstruct(input_image, run_id="failed-detection")
    assert (output_root / "failed-detection" / "input.png").is_file()


def test_run_id_cannot_escape_output_root(input_image: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="run_id"):
        Reconstructor(
            backend=FakeBackend(), renderer=FakeRenderer(), output_root=tmp_path / "outputs"
        ).reconstruct(input_image, full_image=True, run_id="../escape")


def test_parameter_mesh_mismatch_is_rejected(input_image: Path, tmp_path: Path):
    class MismatchedBackend(FakeBackend):
        def predict(self, rgb, bbox):
            prediction = super().predict(rgb, bbox)
            prediction.reference_vertices = prediction.reference_vertices + np.float32(0.01)
            return prediction

    with pytest.raises(ReconstructionError, match="differs from the model output"):
        Reconstructor(
            backend=MismatchedBackend(),
            renderer=FakeRenderer(),
            output_root=tmp_path / "outputs",
        ).reconstruct(input_image, full_image=True, run_id="mesh-mismatch")
