import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import zero_parameters

from occluded_sitting_smplx.export import make_archive, save_metadata, save_obj, save_parameters
from occluded_sitting_smplx.types import BBox


def test_parameter_shapes_and_finite_values():
    parameters = zero_parameters()
    assert parameters.body_pose.shape == (21, 3)
    assert parameters.left_hand_pose.shape == (15, 3)
    parameters.body_pose[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        type(parameters)(**parameters.as_dict())


def test_bbox_validation():
    assert BBox.from_sequence([1, 2, 3, 4]).area == 12
    with pytest.raises(ValueError, match="bbox"):
        BBox.from_sequence([1, 2, 3])


def test_export_bundle(tmp_path: Path):
    obj = tmp_path / "mesh.obj"
    params = tmp_path / "params.npz"
    metadata = tmp_path / "metadata.json"
    archive = tmp_path / "result.zip"
    save_obj(
        obj,
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    save_parameters(params, zero_parameters())
    save_metadata(metadata, {"schema_version": "1.0", "unicode": "坐姿"})
    make_archive(archive, [obj, params, metadata])

    assert obj.read_text(encoding="utf-8").splitlines()[-1] == "f 1 2 3"
    with np.load(params) as values:
        assert values["body_pose"].shape == (21, 3)
    assert json.loads(metadata.read_text(encoding="utf-8"))["unicode"] == "坐姿"
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"mesh.obj", "params.npz", "metadata.json"}
