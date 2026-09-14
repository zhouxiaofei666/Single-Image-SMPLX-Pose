"""Locations and validation for upstream source, checkpoints, and licensed body models."""

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .errors import AssetError

SMPLERX_COMMIT = "064baef0e4ab5277a3297691bc1d46ea5412586f"
SMPLERX_CHECKPOINT = "smpler_x_s32.pth.tar"
DETECTOR_CHECKPOINT = "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
DETECTOR_CONFIG = "mmdet_faster_rcnn_r50_fpn_coco.py"

POSE_ONLY_SMPL_FILE = "SMPL_NEUTRAL.pkl"
POSE_ONLY_SMPLX_FILE = "SMPLX_NEUTRAL.npz"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AssetPaths:
    root: Path
    upstream: Path
    checkpoint: Path
    detector_checkpoint: Path
    detector_config: Path
    body_models: Path
    runtime_body_models: Path

    @classmethod
    def defaults(cls, root: Path = None) -> "AssetPaths":
        resolved_root = (root or project_root()).resolve()
        models = resolved_root / "models"
        return cls(
            root=resolved_root,
            upstream=resolved_root / "third_party" / "SMPLer-X",
            checkpoint=models / "checkpoints" / SMPLERX_CHECKPOINT,
            detector_checkpoint=models / "detectors" / DETECTOR_CHECKPOINT,
            detector_config=models / "detectors" / DETECTOR_CONFIG,
            body_models=models / "body_models",
            runtime_body_models=models / "runtime_body_models",
        )

    def required(self, include_detector: bool = True) -> Dict[str, Path]:
        required = {
            "SMPLer-X submodule": self.upstream / "main" / "SMPLer_X.py",
            "SMPLer-X-S32 checkpoint": self.checkpoint,
        }
        if include_detector:
            required.update(
                {
                    "Faster R-CNN checkpoint": self.detector_checkpoint,
                    "Faster R-CNN config": self.detector_config,
                }
            )
        required["SMPL/SMPL_NEUTRAL.pkl"] = (
            self.body_models / "smpl" / POSE_ONLY_SMPL_FILE
        )
        required["SMPL-X/SMPLX_NEUTRAL.npz"] = (
            self.body_models / "smplx" / POSE_ONLY_SMPLX_FILE
        )
        return required

    def missing(self, include_detector: bool = True) -> Dict[str, Path]:
        return {
            label: path
            for label, path in self.required(include_detector=include_detector).items()
            if not path.is_file()
        }

    def assert_ready(self, include_detector: bool = True) -> None:
        missing = self.missing(include_detector=include_detector)
        if missing:
            formatted = "\n".join(f"  - {label}: {path}" for label, path in missing.items())
            raise AssetError(
                "Required assets are missing:\n"
                f"{formatted}\n"
                "Run scripts/download_assets.py for public weights, then follow README.md "
                "to place the licensed SMPL/SMPL-X files."
            )

    def ensure_public_directories(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.detector_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        (self.body_models / "smpl").mkdir(parents=True, exist_ok=True)
        (self.body_models / "smplx").mkdir(parents=True, exist_ok=True)

    def prepare_pose_only_runtime_models(self) -> Path:
        """Build an ignored neutral-only compatibility layout for upstream imports.

        SMPLer-X eagerly constructs gendered SMPL/SMPL-X models and loads three
        training/evaluation lookup files even though inference uses none of them.
        Pose-only mode aliases the user-provided neutral models and creates empty
        lookup placeholders, avoiding unnecessary licensed downloads.
        """
        self.assert_ready(include_detector=False)
        smpl_dir = self.runtime_body_models / "smpl"
        smplx_dir = self.runtime_body_models / "smplx"
        smpl_dir.mkdir(parents=True, exist_ok=True)
        smplx_dir.mkdir(parents=True, exist_ok=True)

        neutral_smpl = self.body_models / "smpl" / POSE_ONLY_SMPL_FILE
        for gender in ("NEUTRAL", "MALE", "FEMALE"):
            destination = smpl_dir / f"SMPL_{gender}.pkl"
            if (
                not destination.is_file()
                or destination.stat().st_size != neutral_smpl.stat().st_size
            ):
                shutil.copy2(neutral_smpl, destination)

        neutral_smplx = self.body_models / "smplx" / POSE_ONLY_SMPLX_FILE
        for gender in ("NEUTRAL", "MALE", "FEMALE"):
            destination = smplx_dir / f"SMPLX_{gender}.npz"
            if (
                not destination.is_file()
                or destination.stat().st_size != neutral_smplx.stat().st_size
            ):
                shutil.copy2(neutral_smplx, destination)

        pickle_placeholders = {
            "SMPLX_to_J14.pkl": None,
            "MANO_SMPLX_vertex_ids.pkl": {},
        }
        for name, value in pickle_placeholders.items():
            destination = smplx_dir / name
            if not destination.is_file():
                with destination.open("wb") as handle:
                    pickle.dump(value, handle)
        flame_ids = smplx_dir / "SMPL-X__FLAME_vertex_ids.npy"
        if not flame_ids.is_file():
            import numpy as np

            np.save(flame_ids, np.empty((0,), dtype=np.int64))
        return self.runtime_body_models


def format_asset_report(paths: AssetPaths, include_detector: bool = True) -> str:
    rows: List[str] = []
    for label, path in paths.required(include_detector=include_detector).items():
        rows.append(f"[{'OK' if path.is_file() else 'MISSING'}] {label}: {path}")
    return "\n".join(rows)


def all_parent_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
