"""Opt-in real-model smoke test.

Run manually with:
  SMPLX_TEST_IMAGE=/path/to/person.jpg pytest -m gpu tests/test_gpu_integration.py
"""

import os
from pathlib import Path

import pytest

from occluded_sitting_smplx.reconstructor import Reconstructor


@pytest.mark.gpu
def test_real_smpler_x_s32(tmp_path: Path):
    image = os.environ.get("SMPLX_TEST_IMAGE")
    if not image:
        pytest.skip("Set SMPLX_TEST_IMAGE to run the licensed GPU integration test")
    result = Reconstructor(output_root=tmp_path).reconstruct(Path(image))
    assert result.vertices.shape == (10475, 3)
    assert result.obj_path.is_file()
    assert result.archive_path.is_file()
