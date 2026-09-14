"""End-to-end orchestration shared by the public Python API, CLI, and Web UI."""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .assets import AssetPaths, project_root
from .backend import ReconstructionBackend, SMPLerXBackend
from .errors import ReconstructionError
from .export import make_archive, save_metadata, save_obj, save_parameters
from .image import choose_largest_bbox, full_image_bbox, load_normalized_rgb, save_rgb_png
from .render import MeshRenderer
from .types import BBox, ReconstructionResult


class Reconstructor:
    """Reconstruct one primary person from a single RGB image."""

    def __init__(
        self,
        backend: Optional[ReconstructionBackend] = None,
        renderer: Optional[MeshRenderer] = None,
        output_root: Optional[Path] = None,
    ) -> None:
        self.backend = backend or SMPLerXBackend(AssetPaths.defaults())
        self.renderer = renderer or MeshRenderer()
        self.output_root = Path(output_root or project_root() / "outputs").resolve()

    def reconstruct(
        self,
        image_path: Union[str, Path],
        bbox: Optional[Union[BBox, Sequence[float]]] = None,
        full_image: bool = False,
        run_id: Optional[str] = None,
    ) -> ReconstructionResult:
        if bbox is not None and full_image:
            raise ValueError("bbox and full_image are mutually exclusive")
        started = time.perf_counter()
        rgb = load_normalized_rgb(Path(image_path))
        load_seconds = time.perf_counter() - started
        height, width = rgb.shape[:2]

        safe_run_id = run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        if not safe_run_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores")
        run_dir = (self.output_root / safe_run_id).resolve()
        if run_dir.parent != self.output_root:
            raise ValueError("run_id resolved outside output_root")
        run_dir.mkdir(parents=True, exist_ok=False)
        normalized_path = run_dir / "input.png"
        save_rgb_png(rgb, normalized_path)

        detection_started = time.perf_counter()
        if bbox is not None:
            selected_bbox = bbox if isinstance(bbox, BBox) else BBox.from_sequence(bbox)
            selected_bbox = selected_bbox.clamp(width, height)
            selection_mode = "explicit_bbox"
        elif full_image:
            selected_bbox = full_image_bbox(rgb)
            selection_mode = "full_image"
        else:
            selected_bbox = choose_largest_bbox(self.backend.detect(rgb), rgb)
            selection_mode = "auto_largest_person"
        detection_seconds = time.perf_counter() - detection_started

        inference_started = time.perf_counter()
        prediction = self.backend.predict(rgb, selected_bbox)
        vertices = np.asarray(self.backend.build_mesh(prediction.parameters), dtype=np.float32)
        faces = np.asarray(self.backend.faces, dtype=np.int32)
        inference_seconds = time.perf_counter() - inference_started
        if vertices.shape != (10475, 3):
            raise ReconstructionError(f"Expected 10475 SMPL-X vertices, got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ReconstructionError(f"Invalid SMPL-X face array: {faces.shape}")
        regeneration_error = None
        if prediction.reference_vertices is not None:
            reference = np.asarray(prediction.reference_vertices, dtype=np.float32)
            if reference.shape != vertices.shape:
                raise ReconstructionError(
                    "Reference mesh has unexpected shape "
                    f"{reference.shape}; expected {vertices.shape}"
                )
            regeneration_error = float(np.max(np.abs(reference - vertices)))
            if regeneration_error > 1e-4:
                raise ReconstructionError(
                    "Mesh regenerated from exported parameters differs from the model output "
                    f"(maximum absolute error {regeneration_error:.6g})."
                )

        obj_path = run_dir / "mesh.obj"
        parameters_path = run_dir / "smplx_params.npz"
        overlay_path = run_dir / "overlay.png"
        preview_path = run_dir / "mesh_preview.png"
        metadata_path = run_dir / "metadata.json"
        archive_path = run_dir / "result.zip"

        export_started = time.perf_counter()
        save_obj(obj_path, vertices, faces)
        save_parameters(parameters_path, prediction.parameters)
        self.renderer.overlay(rgb, vertices, faces, prediction.camera, overlay_path)
        self.renderer.preview(vertices, faces, preview_path)
        export_seconds = time.perf_counter() - export_started
        timings = {
            "image_load": round(load_seconds, 6),
            "person_selection": round(detection_seconds, 6),
            "inference_and_mesh": round(inference_seconds, 6),
            "render_and_export": round(export_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        }
        peak_memory = getattr(self.backend, "peak_gpu_memory_mb", None)
        metadata = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "name": self.backend.model_name,
                "upstream_commit": self.backend.model_commit,
                "training_performed": False,
                "detail_policy": {
                    "body_pose": "predicted",
                    "hand_pose": "predicted",
                    "face_expression": "neutralized",
                    "jaw_and_eye_pose": "neutralized",
                },
            },
            "image": {"width": width, "height": height, "normalized_file": "input.png"},
            "person_selection": {
                "mode": selection_mode,
                "requested_bbox_xywh": selected_bbox.as_xywh(),
                "processed_bbox_xywh": prediction.processed_bbox.as_xywh(),
            },
            "camera": prediction.camera.as_dict(),
            "mesh": {
                "coordinate_system": "SMPL-X camera coordinates",
                "units": "meters",
                "vertex_count": int(len(vertices)),
                "face_count": int(len(faces)),
                "regeneration_max_abs_error": regeneration_error,
            },
            "parameters": prediction.parameters.shape_summary(),
            "runtime": {
                "timings_seconds": timings,
                "peak_gpu_memory_mb": peak_memory,
            },
        }
        save_metadata(metadata_path, metadata)
        make_archive(
            archive_path,
            [
                normalized_path,
                obj_path,
                parameters_path,
                metadata_path,
                overlay_path,
                preview_path,
            ],
        )
        return ReconstructionResult(
            run_dir=run_dir,
            parameters=prediction.parameters,
            vertices=vertices,
            faces=faces,
            bbox=prediction.processed_bbox,
            camera=prediction.camera,
            source_image=normalized_path,
            obj_path=obj_path,
            parameters_path=parameters_path,
            metadata_path=metadata_path,
            overlay_path=overlay_path,
            preview_path=preview_path,
            archive_path=archive_path,
            timings=timings,
        )
