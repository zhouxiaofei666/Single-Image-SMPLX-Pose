"""Thin runtime adapter around the pinned official SMPLer-X implementation."""

import sys
from collections import OrderedDict
from typing import List, Optional, Protocol

import numpy as np

from .assets import SMPLERX_COMMIT, AssetPaths
from .errors import AssetError, ReconstructionError
from .types import BackendPrediction, BBox, CameraParameters, SMPLXParameters


class ReconstructionBackend(Protocol):
    model_name: str
    model_commit: str

    def detect(self, rgb: np.ndarray) -> List[BBox]: ...

    def predict(self, rgb: np.ndarray, bbox: BBox) -> BackendPrediction: ...

    def build_mesh(self, parameters: SMPLXParameters) -> np.ndarray: ...

    @property
    def faces(self) -> np.ndarray: ...


class SMPLerXBackend:
    """Load-once adapter for the official pretrained SMPLer-X-S32 model.

    SMPLer-X currently contains CUDA-specific calls, so the reconstruction device
    is intentionally restricted to CUDA. Detection can stay on CPU to preserve VRAM.
    """

    model_name = "SMPLer-X-S32"
    model_commit = SMPLERX_COMMIT

    def __init__(
        self,
        assets: Optional[AssetPaths] = None,
        device: str = "cuda:0",
        detector_device: str = "cpu",
    ) -> None:
        self.assets = assets or AssetPaths.defaults()
        self.device = device
        self.detector_device = detector_device
        self._model = None
        self._detector = None
        self._torch = None
        self._cfg = None
        self._smpl_x = None
        self._generate_patch_image = None
        self._process_bbox = None

    def _configure_upstream_imports(self) -> None:
        main_dir = self.assets.upstream / "main"
        common_dir = self.assets.upstream / "common"
        transformer_dir = main_dir / "transformer_utils"
        for entry in (str(common_dir), str(main_dir), str(transformer_dir)):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    @staticmethod
    def _safe_rot6d_to_axis_angle(x):
        """Convert Zhou et al. 6D rotations without torchgeometry's bool-mask bug."""
        import torch
        from torch.nn import functional as functional

        rotations = x.reshape(-1, 3, 2)
        first = functional.normalize(rotations[:, :, 0], dim=1)
        second_raw = rotations[:, :, 1]
        second = functional.normalize(
            second_raw - (first * second_raw).sum(dim=1, keepdim=True) * first,
            dim=1,
        )
        third = torch.cross(first, second, dim=1)
        matrix = torch.stack((first, second, third), dim=-1)

        m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
        m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
        m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
        q_abs = torch.sqrt(
            torch.clamp(
                torch.stack(
                    (
                        1.0 + m00 + m11 + m22,
                        1.0 + m00 - m11 - m22,
                        1.0 - m00 + m11 - m22,
                        1.0 - m00 - m11 + m22,
                    ),
                    dim=1,
                ),
                min=0.0,
            )
        )
        candidates = torch.stack(
            (
                torch.stack((q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01), 1),
                torch.stack((m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20), 1),
                torch.stack((m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m12 + m21), 1),
                torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[:, 3] ** 2), 1),
            ),
            dim=1,
        )
        denominator = 2.0 * torch.clamp(q_abs, min=0.1).unsqueeze(-1)
        candidates = candidates / denominator
        row = torch.arange(len(matrix), device=matrix.device)
        quaternion = candidates[row, q_abs.argmax(dim=1)]
        vector = quaternion[:, 1:]
        vector_norm = torch.linalg.vector_norm(vector, dim=1)
        half_angle = torch.atan2(vector_norm, quaternion[:, 0])
        angle = 2.0 * half_angle
        small = angle.abs() < 1e-6
        scale = torch.empty_like(angle)
        scale[small] = 0.5 - angle[small] ** 2 / 48.0
        scale[~small] = torch.sin(half_angle[~small]) / angle[~small]
        axis_angle = vector / scale.unsqueeze(1)
        return torch.nan_to_num(axis_angle)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self.assets.assert_ready(include_detector=False)
        if not self.device.startswith("cuda"):
            raise ReconstructionError("SMPLer-X-S32 requires a CUDA device in this project.")

        self._configure_upstream_imports()
        try:
            import torch
            from config import cfg

            if not torch.cuda.is_available():
                raise ReconstructionError(
                    "CUDA is not available. Install the CUDA build of PyTorch and verify "
                    "that the NVIDIA GPU is visible."
                )

            config_path = self.assets.upstream / "main" / "config" / "config_smpler_x_s32.py"
            cfg.get_config_fromfile(str(config_path))
            cfg.human_model_path = str(self.assets.prepare_pose_only_runtime_models())
            cfg.encoder_config_file = str(
                self.assets.upstream
                / "main"
                / "transformer_utils"
                / "configs"
                / "smpler_x"
                / "encoder"
                / "body_encoder_small.py"
            )
            cfg.pretrained_model_path = str(self.assets.checkpoint)
            cfg.num_gpus = 1
            cfg.testset = "EHF"

            import SMPLer_X as smpler_x_module
            from utils.human_models import smpl_x
            from utils.preprocessing import generate_patch_image, process_bbox

            smpler_x_module.rot6d_to_axis_angle = self._safe_rot6d_to_axis_angle

            cuda_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            torch.cuda.set_device(cuda_index)
            model = torch.nn.DataParallel(
                smpler_x_module.get_model("test"), device_ids=[cuda_index]
            ).cuda(cuda_index)
            checkpoint = torch.load(str(self.assets.checkpoint), map_location=self.device)
            state_dict = OrderedDict()
            for key, value in checkpoint["network"].items():
                if not key.startswith("module."):
                    key = "module." + key
                key = key.replace("module.backbone", "module.encoder")
                key = key.replace("body_rotation_net", "body_regressor")
                key = key.replace("hand_rotation_net", "hand_regressor")
                state_dict[key] = value
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_network_keys = [
                key for key in incompatible.missing_keys if "smplx_layer" not in key
            ]
            if missing_network_keys:
                raise ReconstructionError(
                    "The SMPLer-X checkpoint is incompatible; missing keys: "
                    + ", ".join(missing_network_keys[:5])
                )
            model.eval()
        except ReconstructionError:
            raise
        except Exception as exc:
            raise AssetError(
                "Failed to initialize SMPLer-X. Verify the Python environment, submodule, "
                "checkpoint, and licensed body-model files."
            ) from exc

        self._torch = torch
        self._cfg = cfg
        self._model = model
        self._smpl_x = smpl_x
        self._generate_patch_image = generate_patch_image
        self._process_bbox = process_bbox

    def _load_detector(self) -> None:
        if self._detector is not None:
            return
        missing = {
            label: path
            for label, path in {
                "Faster R-CNN checkpoint": self.assets.detector_checkpoint,
                "Faster R-CNN config": self.assets.detector_config,
            }.items()
            if not path.is_file()
        }
        if missing:
            detail = "\n".join(f"  - {name}: {path}" for name, path in missing.items())
            raise AssetError(f"Person-detector assets are missing:\n{detail}")
        self._configure_upstream_imports()
        try:
            from mmdet.apis import init_detector

            self._detector = init_detector(
                str(self.assets.detector_config),
                str(self.assets.detector_checkpoint),
                device=self.detector_device,
            )
        except Exception as exc:
            raise AssetError("Failed to initialize the MMDetection person detector.") from exc

    def detect(self, rgb: np.ndarray) -> List[BBox]:
        self._load_detector()
        try:
            from mmdet.apis import inference_detector

            result = inference_detector(self._detector, rgb[:, :, ::-1].copy())
            person_rows = result[0] if isinstance(result, (list, tuple)) else result
            boxes = []
            for row in np.asarray(person_rows):
                x1, y1, x2, y2 = [float(value) for value in row[:4]]
                score = float(row[4]) if len(row) > 4 else None
                if score is not None and score < 0.5:
                    continue
                boxes.append(BBox(x1, y1, x2 - x1, y2 - y1, score))
            return boxes
        except Exception as exc:
            raise ReconstructionError("Person detection failed.") from exc

    def predict(self, rgb: np.ndarray, bbox: BBox) -> BackendPrediction:
        self._load_model()
        torch = self._torch
        cfg = self._cfg
        height, width = rgb.shape[:2]
        processed = self._process_bbox(
            np.asarray(bbox.as_xywh(), dtype=np.float32), width, height
        )
        if processed is None:
            raise ReconstructionError("The selected person box is invalid after preprocessing.")
        patch, _, _ = self._generate_patch_image(
            rgb.astype(np.float32), processed, 1.0, 0.0, False, cfg.input_img_shape
        )
        tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().div_(255.0)
        tensor = tensor.unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                output = self._model({"img": tensor}, {}, {}, "test")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise ReconstructionError(
                    "CUDA ran out of memory. Close other GPU applications and retry with the "
                    "detector on CPU."
                ) from exc
            raise

        def array(name: str, shape) -> np.ndarray:
            return output[name].detach().cpu().numpy().astype(np.float32).reshape(shape)

        parameters = SMPLXParameters(
            global_orient=array("smplx_root_pose", (1, 3)),
            body_pose=array("smplx_body_pose", (21, 3)),
            left_hand_pose=array("smplx_lhand_pose", (15, 3)),
            right_hand_pose=array("smplx_rhand_pose", (15, 3)),
            jaw_pose=np.zeros((1, 3), dtype=np.float32),
            leye_pose=np.zeros((1, 3), dtype=np.float32),
            reye_pose=np.zeros((1, 3), dtype=np.float32),
            betas=array("smplx_shape", (1, 10)),
            expression=np.zeros((1, 10), dtype=np.float32),
            transl=array("cam_trans", (1, 3)),
        )
        focal = (
            float(cfg.focal[0] / cfg.input_body_shape[1] * processed[2]),
            float(cfg.focal[1] / cfg.input_body_shape[0] * processed[3]),
        )
        principal_point = (
            float(cfg.princpt[0] / cfg.input_body_shape[1] * processed[2] + processed[0]),
            float(cfg.princpt[1] / cfg.input_body_shape[0] * processed[3] + processed[1]),
        )
        return BackendPrediction(
            parameters=parameters,
            camera=CameraParameters(focal=focal, principal_point=principal_point),
            processed_bbox=BBox.from_sequence(processed.tolist()),
            # Upstream mesh output contains predicted face/jaw/eye details. This
            # project neutralizes them, so compare with the neutral parameter mesh.
            reference_vertices=self.build_mesh(parameters),
        )

    def build_mesh(self, parameters: SMPLXParameters) -> np.ndarray:
        self._load_model()
        torch = self._torch
        values = {
            key: torch.from_numpy(value).float()
            for key, value in parameters.as_dict().items()
        }
        with torch.inference_mode():
            output = self._smpl_x.layer["neutral"](
                betas=values["betas"],
                body_pose=values["body_pose"].reshape(1, -1),
                global_orient=values["global_orient"],
                left_hand_pose=values["left_hand_pose"].reshape(1, -1),
                right_hand_pose=values["right_hand_pose"].reshape(1, -1),
                jaw_pose=values["jaw_pose"],
                leye_pose=values["leye_pose"],
                reye_pose=values["reye_pose"],
                expression=values["expression"],
                transl=values["transl"],
            )
        vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
        if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
            raise ReconstructionError(f"Unexpected SMPL-X vertex array: {vertices.shape}")
        return vertices

    @property
    def faces(self) -> np.ndarray:
        self._load_model()
        return np.asarray(self._smpl_x.face, dtype=np.int32)

    @property
    def peak_gpu_memory_mb(self) -> Optional[float]:
        if self._torch is None or not self._torch.cuda.is_available():
            return None
        return float(self._torch.cuda.max_memory_allocated() / (1024**2))
