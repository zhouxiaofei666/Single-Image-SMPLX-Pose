#!/usr/bin/env python3
"""Download only redistributable public checkpoints/configs.

SMPL and SMPL-X model files are intentionally never downloaded by this script.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occluded_sitting_smplx.assets import AssetPaths  # noqa: E402

DOWNLOADS = {
    "checkpoint": (
        "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/"
        "smpler_x_s32.pth.tar?download=true"
    ),
    "detector_checkpoint": (
        "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
        "faster_rcnn_r50_fpn_1x_coco/"
        "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    ),
    "detector_config": (
        "https://raw.githubusercontent.com/openxrlab/xrmocap/main/configs/modules/"
        "human_perception/mmdet_faster_rcnn_r50_fpn_coco.py"
    ),
}

SMPLERX_S32_SHA256 = "1c78569f4e7b6e22e7166b55f322e5d5e019073b8d1a430a6926f93b4c86aa62"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    force: bool = False,
    expected_sha256: str = None,
) -> None:
    if destination.is_file() and not force:
        if expected_sha256 and sha256(destination) != expected_sha256:
            raise RuntimeError(
                f"Existing file has the wrong SHA-256; remove or replace it: {destination}"
            )
        print(f"skip existing: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}\n      -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "Occluded-Sitting-SMPLX/0.1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if expected_sha256 and sha256(temporary) != expected_sha256:
            raise RuntimeError(f"SHA-256 verification failed for {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-detector", action="store_true")
    args = parser.parse_args()
    assets = AssetPaths.defaults(ROOT)
    assets.ensure_public_directories()
    download(
        DOWNLOADS["checkpoint"],
        assets.checkpoint,
        args.force,
        expected_sha256=SMPLERX_S32_SHA256,
    )
    if not args.skip_detector:
        download(
            DOWNLOADS["detector_checkpoint"], assets.detector_checkpoint, args.force
        )
        download(DOWNLOADS["detector_config"], assets.detector_config, args.force)
    print("\nPublic assets are ready. Licensed SMPL/SMPL-X files must be added manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
