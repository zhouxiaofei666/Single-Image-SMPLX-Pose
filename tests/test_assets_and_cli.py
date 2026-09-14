from pathlib import Path

import numpy as np

from occluded_sitting_smplx.assets import AssetPaths, format_asset_report
from occluded_sitting_smplx.cli import build_parser
from occluded_sitting_smplx.types import BBox


def test_asset_report_lists_missing_files(tmp_path: Path):
    assets = AssetPaths.defaults(tmp_path)
    report = format_asset_report(assets)
    assert "[MISSING] SMPLer-X-S32 checkpoint" in report
    assert "SMPL-X/SMPLX_NEUTRAL.npz" in report
    assert "SMPLX_MALE" not in report


def test_cli_bbox_parsing():
    args = build_parser().parse_args(
        ["infer", "person.png", "--bbox", "10,20,100,200", "-o", "result"]
    )
    assert args.bbox == BBox(10, 20, 100, 200)
    assert args.output == Path("result")


def test_pose_only_runtime_layout_uses_neutral_models(tmp_path: Path):
    assets = AssetPaths.defaults(tmp_path)
    assets.ensure_public_directories()
    (assets.upstream / "main").mkdir(parents=True)
    (assets.upstream / "main" / "SMPLer_X.py").write_text("# test", encoding="utf-8")
    assets.checkpoint.write_bytes(b"checkpoint")
    neutral_smpl = assets.body_models / "smpl" / "SMPL_NEUTRAL.pkl"
    neutral_smplx = assets.body_models / "smplx" / "SMPLX_NEUTRAL.npz"
    neutral_smpl.write_bytes(b"neutral-smpl")
    neutral_smplx.write_bytes(b"neutral-smplx")

    runtime = assets.prepare_pose_only_runtime_models()

    assert (runtime / "smpl" / "SMPL_MALE.pkl").read_bytes() == b"neutral-smpl"
    assert (runtime / "smplx" / "SMPLX_FEMALE.npz").read_bytes() == b"neutral-smplx"
    assert np.load(runtime / "smplx" / "SMPL-X__FLAME_vertex_ids.npy").size == 0
