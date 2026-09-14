# Single-Image-SMPLX-Pose

[![CI](https://github.com/zhouxiaofei666/Single-Image-SMPLX-Pose/actions/workflows/ci.yml/badge.svg)](https://github.com/zhouxiaofei666/Single-Image-SMPLX-Pose/actions/workflows/ci.yml)

> Local single-image SMPL-X pose and mesh reconstruction powered by the official pretrained **SMPLer-X-S32** model, with OBJ export, image overlays, multi-view rendering, and a Gradio interface.

**Single-Image-SMPLX-Pose** is a research-oriented local tool for reconstructing a 3D SMPL-X human mesh from a single RGB image. It uses the official pretrained [SMPLer-X](https://github.com/MotrixLab/SMPLer-X) **S32** checkpoint directly and does **not** train or fine-tune a model.

The project focuses on a clear, inspectable inference pipeline for ordinary single-person images, with particular interest in **sitting poses** and **partial occlusion**. Body pose, both hands, body shape, and camera parameters are retained. Predicted facial expression, jaw pose, and eye pose are intentionally discarded and replaced with neutral values.

```text
RGB image
   ↓
person selection / crop
   ↓
pretrained SMPLer-X-S32
   ↓
SMPL-X pose + shape + camera parameters
   ↓
neutralized SMPL-X mesh
   ↓
OBJ / NPZ / metadata / overlay / multi-view preview / ZIP
```

## Project scope

This repository is both a **learning project** and a **research prototype**. Its goal is to make a complete 3D human reconstruction workflow easy to inspect, run, and modify:

- convert a single RGB image into structured 3D human pose parameters;
- understand the relationship between SMPL-X parameters, vertices, faces, and OBJ meshes;
- integrate a pretrained model with a Python API, CLI, and local Gradio interface;
- study how sitting posture, occlusion, viewpoint, and monocular depth ambiguity affect reconstruction.

This project does **not** claim to introduce a new human-mesh model or outperform the original SMPLer-X method. It is also not intended for medical measurement, production-grade motion capture, biometric identification, or metric-scale recovery.

A single image cannot uniquely determine the true depth, physical scale, back-side pose, or fully occluded limbs. Outputs should therefore be treated as **model-based estimates** for visualization, analysis, and downstream research.

> **License note:** SMPLer-X is distributed under the S-Lab License 1.0 for non-commercial use. SMPL and SMPL-X body-model files must be downloaded separately after accepting their official licenses. This repository and its setup scripts do not redistribute those restricted files.

## Features

- Single-image input: JPG, PNG, WebP, and BMP.
- EXIF orientation correction before inference.
- Automatic Faster R-CNN person detection with largest-person selection.
- Optional full-image / manually cropped person mode when automatic detection is unsuitable.
- SMPL-X body pose, both hands, body shape, translation, and camera-parameter export.
- Neutralized face, jaw, and eye parameters for a pose-focused output.
- Regeneration of a neutral **10,475-vertex SMPL-X mesh** from exported parameters.
- OBJ mesh export.
- Mesh overlay on the source image.
- Front, side, and back mesh previews.
- Interactive browser-based 3D mesh viewer.
- Shared inference core for the Python API, CLI, and Gradio Web UI.
- Local-only processing by default (`share=False`).

## Supported environment

The primary target is **Windows 10/11 with an NVIDIA GPU**. A WSL2 Ubuntu 20.04/22.04 setup path is also included.

The current configuration was designed with a **6 GB RTX 2060** in mind:

- person detection runs on CPU by default;
- SMPLer-X inference runs on CUDA;
- requests are processed one at a time to reduce VRAM pressure.

Requirements:

- NVIDIA driver with a working `nvidia-smi`;
- Python 3.9 on native Windows;
- Miniconda/Anaconda for the WSL2 setup path;
- Git with submodule support;
- roughly 8 GB of free disk space for the environment, public checkpoints, and licensed body-model files.

The upstream SMPLer-X implementation contains direct `.cuda()` calls, so this project currently does **not** support full CPU-only reconstruction.

The native Windows environment is intentionally pinned to:

- PyTorch 1.12.1 + CUDA 11.3;
- MMCV 1.7.1;
- compatible OpenMMLab dependencies.

Do not upgrade the same environment to MMCV 2.x or MMDetection 3.x unless you also update the upstream integration code.

## Installation

### 1. Clone the repository and pinned upstream submodule

```bash
git clone --recurse-submodules https://github.com/zhouxiaofei666/Single-Image-SMPLX-Pose.git
cd Single-Image-SMPLX-Pose
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The SMPLer-X submodule is pinned to commit:

```text
064baef0e4ab5277a3297691bc1d46ea5412586f
```

This keeps the local adapter aligned with the upstream interface it was written against.

### 2. Create the Windows environment and download public assets

Run in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

The script creates a project-local `.venv`, installs the pinned CUDA/PyTorch/OpenMMLab stack, installs this project with Web UI support, installs the upstream dependencies, and downloads the public checkpoints.

If the public checkpoints already exist:

```powershell
.\scripts\setup_windows.ps1 -SkipAssets
```

WSL2 alternative:

```bash
bash scripts/setup_wsl.sh
conda activate occluded-sitting-smplx
```

The WSL2 script uses Python 3.8 and installs OSMesa for headless rendering.

The setup scripts download these public files:

```text
models/checkpoints/smpler_x_s32.pth.tar
models/detectors/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
models/detectors/mmdet_faster_rcnn_r50_fpn_coco.py
```

You can also download only the public assets with:

```bash
python scripts/download_assets.py
```

### 3. Add the licensed neutral body models

Register on the official [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) websites, read the applicable terms, accept the licenses, and download the required files.

Pose-focused inference only requires the two neutral model files below:

```text
models/body_models/
├── smpl/
│   └── SMPL_NEUTRAL.pkl
└── smplx/
    └── SMPLX_NEUTRAL.npz
```

The upstream implementation eagerly loads several gender-specific models and training/evaluation lookup files even though they are not used by this inference workflow. At runtime, this project creates an ignored compatibility layout under:

```text
models/runtime_body_models/
```

For those **unused compatibility entries**, the project aliases the user-provided neutral files and creates empty lookup placeholders. It does not fabricate or download additional licensed assets, and the actual pose reconstruction still uses the neutral SMPL-X layer.

Both the licensed model directory and the runtime compatibility directory are excluded by `.gitignore`.

Verify the installation with:

```bash
smplx-reconstruct check-assets
# or
python scripts/check_assets.py
```

## Usage

### Web UI

Start the local Gradio application:

```bash
smplx-reconstruct serve
# equivalent to: python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The interface provides two person-selection modes:

- **Automatically detect the largest person** — recommended for images containing one clearly visible person.
- **Use the edited image as the full person crop** — useful after manually cropping the upload when the image contains multiple people, a sitting pose is only partially detected, or automatic detection fails.

The server listens on the local machine and launches with `share=False`. Images are not sent to a hosted inference service by this application.

### CLI

Automatically detect the primary person:

```bash
smplx-reconstruct infer /path/to/person.jpg -o outputs
```

Treat the full image as the person crop:

```bash
smplx-reconstruct infer person.png -o outputs --full-image
```

Provide an explicit bounding box in source-image pixels as `x,y,width,height`:

```bash
smplx-reconstruct infer person.png -o outputs --bbox 120,40,760,980
```

Defaults:

```text
--device cuda:0
--detector-device cpu
```

Each inference creates a unique run directory under the selected output root so previous results are not overwritten.

### Python API

```python
from occluded_sitting_smplx import BBox, Reconstructor

reconstructor = Reconstructor(output_root="outputs")
result = reconstructor.reconstruct(
    "person.jpg",
    bbox=BBox(x=120, y=40, width=760, height=980),
)

print(result.obj_path)
print(result.parameters.body_pose.shape)  # (21, 3)
```

## Output structure

Each run directory contains:

| File | Description |
|---|---|
| `input.png` | EXIF-corrected RGB input image |
| `mesh.obj` | SMPL-X triangle mesh in metric camera coordinates |
| `smplx_params.npz` | SMPL-X-compatible parameters that regenerate the pose mesh |
| `metadata.json` | Schema, model version, person box, camera, timing, and GPU-memory metadata |
| `overlay.png` | Mesh projection over the input image |
| `mesh_preview.png` | Independent front, side, and back renders |
| `result.zip` | Archive containing the exported result files |

### `smplx_params.npz`

All arrays are stored as `float32`.

| Field | Shape | Meaning |
|---|---:|---|
| `global_orient` | `(1, 3)` | Root axis-angle rotation |
| `body_pose` | `(21, 3)` | Body-joint axis-angle rotations |
| `left_hand_pose`, `right_hand_pose` | `(15, 3)` | Full hand axis-angle rotations |
| `jaw_pose`, `leye_pose`, `reye_pose` | `(1, 3)` | Neutral compatibility fields fixed to zero |
| `betas` | `(1, 10)` | Body shape |
| `expression` | `(1, 10)` | Fixed to zero; predicted facial expression is not retained |
| `transl` | `(1, 3)` | Camera-coordinate translation in meters |

## Sitting poses and occlusion

For sitting-pose or partially occluded images:

- keep the head, torso, pelvis, and as much of the visible legs as possible;
- avoid cropping away body cues near the seat-contact region;
- if the detector selects the wrong region, crop tightly around the target person and use full-image mode;
- expect larger ambiguity for hidden legs, crossed limbs, loose clothing, extreme perspective, mirrors, and severe person-to-person overlap.

Monocular reconstruction cannot recover a unique true pose for completely invisible body parts. The output reflects the pretrained model's learned prior, not direct observation of hidden geometry.

This repository does not retrain SMPLer-X specifically for sitting or occlusion and therefore does not claim improved accuracy over the original model in those scenarios.

## Troubleshooting

### `CUDA is not available`

Confirm that the GPU is visible:

```bash
nvidia-smi
```

Then check PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### CUDA out of memory

Close other GPU-heavy applications, keep the detector on CPU, and process one image at a time.

The S32 model is the only backend used by this project; the application does not silently switch to a different model with different quality or memory requirements.

### MMCV / MMDetection import errors

Use the project `.venv` on Windows, or the project Conda environment on WSL2, with the versions installed by the provided scripts.

Do not upgrade the environment to MMCV 2.x or MMDetection 3.x without updating the integration code.

### `pyrender` / OSMesa errors

On Windows, use desktop OpenGL, update the NVIDIA driver, and start the application from a normal PowerShell session. Do not set `PYOPENGL_PLATFORM=osmesa` on native Windows.

For WSL2 or headless Linux:

```bash
sudo apt-get install libosmesa6 libgl1-mesa-glx
export PYOPENGL_PLATFORM=osmesa
```

### No person detected

Crop the image around the target person in the Web UI and choose **Use the edited image as the full person crop**, or use `--full-image` / `--bbox` from the CLI.

## Development and validation

Install the development extras and run the tests that do not require model weights or a GPU:

```bash
python -m pip install -e ".[dev]"
ruff check src tests scripts
pytest -m "not gpu"
```

Run the real-model integration test with an image you are authorized to use:

```bash
SMPLX_TEST_IMAGE=/path/to/person.jpg pytest -m gpu tests/test_gpu_integration.py
```

CI runs the lightweight backend tests and does not download restricted model assets. The real-model integration test also records PyTorch peak GPU memory in `metadata.json`, which is useful when checking compatibility with a 6 GB GPU.

## License and citation

The repository code follows the **S-Lab License 1.0** used by SMPLer-X and is limited to non-commercial use. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

If this repository contributes to published research, please cite the original SMPLer-X and SMPL-X works:

```bibtex
@inproceedings{cai2023smplerx,
  title={{SMPLer-X}: Scaling up expressive human pose and shape estimation},
  author={Cai, Zhongang and Yin, Wanqi and Zeng, Ailing and Wei, Chen and
          Sun, Qingping and Wang, Yanjun and Pang, Hui En and Mei, Haiyi and
          Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yang, Lei and
          Liu, Ziwei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@inproceedings{SMPL-X:2019,
  title={Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author={Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and
          Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and
          Black, Michael J.},
  booktitle={CVPR},
  year={2019}
}
```

## Repository hygiene

Before publishing changes or creating a fork, make sure the repository does not contain personal images, licensed body-model files, model checkpoints, generated outputs, or a local virtual environment.

The following are already excluded by `.gitignore`:

```text
models/body_models/
models/checkpoints/
outputs/
.venv/
```

Useful checks on Windows:

```powershell
git status --short
git ls-files | Select-String '\.(pth|pt|tar|pkl|npz|jpg|jpeg|png|zip)$'
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe -m pytest -m "not gpu"
```

After making changes:

```powershell
git add .
git commit -m "Describe your change"
git push
```

Contributions, bug reports, and reproducible failure cases are welcome.