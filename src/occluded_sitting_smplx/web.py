"""Local-only Gradio interface."""

from pathlib import Path
from typing import Any, Dict, Optional

from .assets import AssetPaths
from .backend import SMPLerXBackend
from .errors import ReconstructionError
from .reconstructor import Reconstructor

AUTO_DETECT_MODE = "Automatically detect the largest person"
FULL_IMAGE_MODE = "Use the edited image as the full person crop"


def _editor_path(value: Optional[Dict[str, Any]]) -> Optional[str]:
    if not value:
        return None
    return value.get("composite") or value.get("background")


def build_demo(reconstructor: Reconstructor):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the Web UI with: pip install -e '.[web]'") from exc

    def run(editor_value, selection_mode):
        image_path = _editor_path(editor_value)
        if not image_path:
            return None, None, None, {}, None, "Please upload an image first."
        try:
            result = reconstructor.reconstruct(
                image_path,
                full_image=selection_mode == FULL_IMAGE_MODE,
            )
        except (ReconstructionError, ValueError) as exc:
            return None, None, None, {}, None, f"Reconstruction failed: {exc}"
        summary = {
            "run_dir": str(result.run_dir),
            "bbox_xywh": result.bbox.as_xywh(),
            "camera": result.camera.as_dict(),
            "pose_parameter_shapes": {
                name: shape
                for name, shape in result.parameters.shape_summary().items()
                if name
                in {
                    "global_orient",
                    "body_pose",
                    "left_hand_pose",
                    "right_hand_pose",
                    "betas",
                    "transl",
                }
            },
            "face_detail": "Neutral (predicted expression, jaw pose, and eye pose are not exported)",
            "timings_seconds": result.timings,
        }
        return (
            str(result.overlay_path),
            str(result.preview_path),
            str(result.obj_path),
            summary,
            str(result.archive_path),
            "Done. The OBJ mesh, SMPL-X parameters, metadata, and renders have been packaged.",
        )

    with gr.Blocks(title="Single-Image SMPL-X Pose", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Single-Image SMPL-X Pose\n"
            "Reconstruct body and hand pose from one RGB image with the pretrained "
            "**SMPLer-X-S32** model. Facial expression, jaw pose, and eye pose are "
            "neutralized. Processing stays on this machine and is not sent to a hosted "
            "inference service."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.ImageEditor(
                    label="Upload or crop a person image",
                    type="filepath",
                    sources=["upload"],
                    height=520,
                )
                mode = gr.Radio(
                    [AUTO_DETECT_MODE, FULL_IMAGE_MODE],
                    value=AUTO_DETECT_MODE,
                    label="Person selection",
                )
                submit = gr.Button("Reconstruct", variant="primary")
                status = gr.Markdown("Waiting for an image.")
                archive = gr.File(label="Download complete result ZIP")
            with gr.Column(scale=1):
                overlay = gr.Image(label="Mesh overlay", type="filepath")
                preview = gr.Image(label="Front / side / back preview", type="filepath")
        with gr.Row():
            model = gr.Model3D(
                label="Interactive 3D mesh",
                display_mode="solid",
                clear_color=(0.96, 0.97, 0.98, 1.0),
                height=600,
            )
            details = gr.JSON(label="Parameters and runtime summary")
        submit.click(
            run,
            inputs=[image, mode],
            outputs=[overlay, preview, model, details, archive, status],
            concurrency_limit=1,
            api_name="reconstruct",
        )
    return demo


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    output_root: Path = Path("outputs"),
    device: str = "cuda:0",
    detector_device: str = "cpu",
) -> None:
    assets = AssetPaths.defaults()
    backend = SMPLerXBackend(
        assets=assets, device=device, detector_device=detector_device
    )
    reconstructor = Reconstructor(backend=backend, output_root=output_root)
    demo = build_demo(reconstructor)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=False,
        # Gradio 4.44's optional /info schema generation is not compatible
        # with newer Pydantic JSON-schema booleans. The UI callback itself
        # remains fully functional and should stay local-only.
        show_api=False,
        allowed_paths=[str(reconstructor.output_root)],
    )
