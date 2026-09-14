"""Local-only Gradio interface."""

from pathlib import Path
from typing import Any, Dict, Optional

from .assets import AssetPaths
from .backend import SMPLerXBackend
from .errors import ReconstructionError
from .reconstructor import Reconstructor


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
            return None, None, None, {}, None, "请先上传图片。"
        try:
            result = reconstructor.reconstruct(
                image_path,
                full_image=selection_mode == "整张编辑结果为人物",
            )
        except (ReconstructionError, ValueError) as exc:
            return None, None, None, {}, None, f"失败：{exc}"
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
            "face_detail": "中性（不导出预测表情、下颌或眼球动作）",
            "timings_seconds": result.timings,
        }
        return (
            str(result.overlay_path),
            str(result.preview_path),
            str(result.obj_path),
            summary,
            str(result.archive_path),
            "完成。OBJ、SMPL-X 参数、元数据和渲染图已打包。",
        )

    with gr.Blocks(title="Occluded-Sitting-SMPLX", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Occluded-Sitting-SMPLX\n"
            "使用预训练 **SMPLer-X-S32** 从单张图片恢复身体与双手姿态；"
            "面部固定为中性。"
            "图片只在本机处理，不会上传到外部服务。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.ImageEditor(
                    label="上传或裁剪人物图片",
                    type="filepath",
                    sources=["upload"],
                    height=520,
                )
                mode = gr.Radio(
                    ["自动检测最大人物", "整张编辑结果为人物"],
                    value="自动检测最大人物",
                    label="人物选择",
                )
                submit = gr.Button("开始重建", variant="primary")
                status = gr.Markdown("等待输入。")
                archive = gr.File(label="下载完整结果 ZIP")
            with gr.Column(scale=1):
                overlay = gr.Image(label="原图 Mesh 叠加", type="filepath")
                preview = gr.Image(label="三视图渲染", type="filepath")
        with gr.Row():
            model = gr.Model3D(
                label="交互式三维网格",
                display_mode="solid",
                clear_color=(0.96, 0.97, 0.98, 1.0),
                height=600,
            )
            details = gr.JSON(label="参数与运行摘要")
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
