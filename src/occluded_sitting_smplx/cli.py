"""Command-line interface for reconstruction, serving, and asset diagnostics."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .assets import AssetPaths, format_asset_report
from .backend import SMPLerXBackend
from .errors import ReconstructionError
from .reconstructor import Reconstructor
from .types import BBox


def _parse_bbox(value: str) -> BBox:
    try:
        parts = [float(item.strip()) for item in value.split(",")]
        return BBox.from_sequence(parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be x,y,width,height") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smplx-reconstruct",
        description="Single-image SMPL-X reconstruction powered by pretrained SMPLer-X-S32.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="reconstruct the primary person in one image")
    infer.add_argument("image", type=Path)
    infer.add_argument("-o", "--output", type=Path, default=Path("outputs"))
    selection = infer.add_mutually_exclusive_group()
    selection.add_argument("--bbox", type=_parse_bbox, help="x,y,width,height in pixels")
    selection.add_argument(
        "--full-image", action="store_true", help="treat the complete image as the person crop"
    )
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--detector-device", default="cpu")

    serve = subparsers.add_parser("serve", help="launch the local Gradio interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--output", type=Path, default=Path("outputs"))
    serve.add_argument("--device", default="cuda:0")
    serve.add_argument("--detector-device", default="cpu")

    check = subparsers.add_parser("check-assets", help="report missing code, weights, and models")
    check.add_argument(
        "--no-detector", action="store_true", help="do not require automatic-detection assets"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-assets":
        assets = AssetPaths.defaults()
        print(format_asset_report(assets, include_detector=not args.no_detector))
        return int(bool(assets.missing(include_detector=not args.no_detector)))

    if args.command == "serve":
        from .web import launch

        launch(
            host=args.host,
            port=args.port,
            output_root=args.output,
            device=args.device,
            detector_device=args.detector_device,
        )
        return 0

    backend = SMPLerXBackend(
        assets=AssetPaths.defaults(),
        device=args.device,
        detector_device=args.detector_device,
    )
    reconstructor = Reconstructor(backend=backend, output_root=args.output)
    try:
        result = reconstructor.reconstruct(
            args.image,
            bbox=args.bbox,
            full_image=args.full_image,
        )
    except (ReconstructionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "mesh": str(result.obj_path),
                "parameters": str(result.parameters_path),
                "overlay": str(result.overlay_path),
                "preview": str(result.preview_path),
                "archive": str(result.archive_path),
                "timings_seconds": result.timings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
