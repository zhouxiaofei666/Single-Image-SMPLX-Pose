"""Deterministic SMPL-X result serialization."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .types import SMPLXParameters


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or infinity")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Occluded-Sitting-SMPLX\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces.astype(np.int64) + 1:
            handle.write(f"f {a} {b} {c}\n")


def save_parameters(path: Path, parameters: SMPLXParameters) -> None:
    np.savez_compressed(path, **parameters.as_dict())


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def make_archive(path: Path, files: Iterable[Path]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)
