"""Headless image rendering and an OBJ-friendly multi-view preview."""

import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .errors import ReconstructionError
from .types import CameraParameters


class MeshRenderer:
    """Render with pyrender, using OSMesa only under Linux/headless WSL."""

    def __init__(self) -> None:
        if platform.system() == "Windows":
            # MMPose 0.28 sets PYOPENGL_PLATFORM=osmesa while importing its
            # optional visualizers. Import pyrender first so PyOpenGL selects
            # the native Windows backend before that upstream side effect.
            os.environ.pop("PYOPENGL_PLATFORM", None)
            self._imports()
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    @staticmethod
    def _imports():
        if platform.system() == "Windows":
            os.environ.pop("PYOPENGL_PLATFORM", None)
        try:
            import pyrender
            import trimesh

            return pyrender, trimesh
        except Exception as exc:
            raise ReconstructionError(
                "3D rendering dependencies failed to load. Check the platform-specific "
                "OpenGL instructions in README.md."
            ) from exc

    def overlay(
        self,
        rgb: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        camera: CameraParameters,
        output_path: Path,
    ) -> None:
        pyrender, trimesh = self._imports()
        mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180.0), [1, 0, 0])
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.75,
            alphaMode="OPAQUE",
            baseColorFactor=(0.88, 0.92, 1.0, 1.0),
        )
        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3, 1.0))
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
        scene.add(
            pyrender.IntrinsicsCamera(
                fx=camera.focal[0],
                fy=camera.focal[1],
                cx=camera.principal_point[0],
                cy=camera.principal_point[1],
            )
        )
        self._add_lights(scene, pyrender)
        height, width = rgb.shape[:2]
        renderer = pyrender.OffscreenRenderer(width, height, point_size=1.0)
        try:
            rendered, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        alpha = (depth > 0).astype(np.float32)[..., None] * 0.88
        blended = rendered[..., :3].astype(np.float32) * alpha + rgb * (1.0 - alpha)
        Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(output_path)

    def preview(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: Path,
        view_size: int = 480,
    ) -> None:
        labels_and_yaws = (("Front", 0.0), ("Side", 90.0), ("Back", 180.0))
        views = [
            self._render_orbit_view(vertices, faces, yaw, view_size)
            for _, yaw in labels_and_yaws
        ]
        canvas = Image.new("RGB", (view_size * len(views), view_size), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)
        for index, ((label, _), view) in enumerate(zip(labels_and_yaws, views)):
            canvas.paste(Image.fromarray(view), (index * view_size, 0))
            draw.rounded_rectangle(
                (index * view_size + 14, 14, index * view_size + 88, 43),
                radius=7,
                fill=(20, 28, 38),
            )
            draw.text((index * view_size + 27, 21), label, fill=(255, 255, 255))
        canvas.save(output_path)

    def _render_orbit_view(
        self, vertices: np.ndarray, faces: np.ndarray, yaw: float, size: int
    ) -> np.ndarray:
        pyrender, trimesh = self._imports()
        centered = vertices.astype(np.float64).copy()
        centered -= (centered.min(axis=0) + centered.max(axis=0)) / 2.0
        extent = max(float(np.ptp(centered, axis=0).max()), 1e-3)
        mesh = trimesh.Trimesh(centered, faces.copy(), process=False)
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180.0), [1, 0, 0])
        )
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0])
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.78,
            baseColorFactor=(0.58, 0.72, 0.92, 1.0),
        )
        scene = pyrender.Scene(
            bg_color=(245, 247, 250, 255), ambient_light=(0.35, 0.35, 0.35, 1.0)
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
        camera_pose = np.eye(4)
        camera_pose[2, 3] = extent * 2.1
        scene.add(pyrender.PerspectiveCamera(yfov=np.radians(42.0)), pose=camera_pose)
        self._add_lights(scene, pyrender)
        renderer = pyrender.OffscreenRenderer(size, size)
        try:
            color, _ = renderer.render(scene)
        finally:
            renderer.delete()
        return color

    @staticmethod
    def _add_lights(scene, pyrender) -> None:
        for translation, intensity in (
            ((0.0, -1.0, 2.0), 2.0),
            ((1.5, 1.0, 2.0), 1.6),
            ((-1.5, 0.5, 1.0), 1.0),
        ):
            pose = np.eye(4)
            pose[:3, 3] = translation
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=pose
            )
