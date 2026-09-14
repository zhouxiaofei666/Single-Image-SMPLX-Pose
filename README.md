# Single-Image-SMPLX-Pose

> 基于官方预训练 SMPLer-X-S32 的单张人体图片 SMPL-X 姿态重建工具，支持 OBJ 导出、二维叠加、多视角渲染和 Gradio 交互查看。

从单张 RGB 人体图片预测身体/双手姿态与 SMPL-X 三维人体网格的本地科研工具。项目直接使用
[SMPLer-X](https://github.com/MotrixLab/SMPLer-X) 官方预训练的 **SMPLer-X-S32**，
不训练、不微调模型。项目只保留姿态重建所需的信息，预测的面部表情、下颌和眼球动作会
被丢弃并固定为中性，面向普通单人图片的姿态重建与可视化，并重点记录坐姿和局部遮挡场景。

## 给初学者的项目说明

这是一个由初学者（作者本人也是刚开始学习这方面的小白）搭建的学习型、科研原型项目，重点是把成熟的开源模型串成一条可以运行的
流程：上传图片 → 检测人物 → 预测 SMPL-X 姿势参数 → 生成 OBJ 网格 → 渲染和导出。项目
没有自行训练人体模型，也不声称在任何场景中超过 SMPLer-X 官方模型。代码中保留了
较完整的注释、命令行接口、Web 界面和测试，方便刚接触深度学习、3D 人体模型或 GitHub
项目的人逐步阅读和运行。

如果代码里还有不成熟的地方，欢迎指出和交流，也请各位大佬多多指教、轻喷；这个项目首先是
一次认真完成的学习记录，而不是声称已经达到工业级水平的成品。

适合用来学习和验证：

- 单张图片如何转换为结构化的 3D 人体姿势；
- SMPL-X 参数、顶点、三角面和 OBJ 文件之间的关系；
- 预训练模型如何通过适配器接入 CLI、Python API 和 Gradio；
- 遮挡、坐姿、视角和单目深度歧义会怎样影响重建结果。

它不是医学测量、商业动作捕捉或真实尺度恢复系统。单张图片无法唯一确定被遮挡肢体的
深度、真实尺寸和背面姿势；结果应作为估计、可视化和后续研究的起点。

> **使用限制：** SMPLer-X 使用 S-Lab License 1.0，仅允许非商业用途；SMPL/SMPL-X
> 模型文件需要分别接受官方网站许可后自行下载。仓库及安装器不会分发这些受限文件。

## 功能

- JPG、PNG、WebP、BMP 单图输入及 EXIF 方向修正。
- Faster R-CNN 自动选择面积最大的人物，或使用手动裁剪后的整张图片。
- SMPL-X 身体与双手姿态、人体形状和相机参数导出；面部始终使用中性参数。
- 从参数重新生成 10475 顶点的中性 SMPL-X 网格并导出 OBJ。
- 原图 Mesh 叠加、正面/侧面/背面渲染和浏览器交互式三维查看。
- 同一推理核心同时服务 Python API、CLI 和本地 Gradio 页面。

<!-- GPU 实机验证后，可把界面截图放入 docs/screenshots/ 并在这里引用。 -->

## 支持环境

默认环境是 **Windows 10/11 + NVIDIA GPU**，也保留 WSL2 Ubuntu 20.04/22.04 方案。已针对
6GB 显存的 RTX 2060 设计：人体检测默认在 CPU，SMPLer-X 在单张、单并发 CUDA 模式运行。

- NVIDIA 驱动与可正常运行的 `nvidia-smi`
- Windows Python 3.9（WSL2 方案使用 Miniconda/Anaconda）
- Git（需支持子模块）
- 约 8GB 磁盘空间用于环境、权重和人体模型

SMPLer-X 上游实现含有直接 `.cuda()` 调用，当前版本不支持纯 CPU 人体重建。Windows 原生
环境固定为 PyTorch 1.12.1/CUDA 11.3 与 MMCV 1.7.1，不要在同一虚拟环境中升级到新版
OpenMMLab 依赖。

## 安装

### 1. 克隆仓库及固定版本的上游代码

```bash
git clone --recurse-submodules https://github.com/zhouxiaofei666/Single-Image-SMPLX-Pose.git
cd Single-Image-SMPLX-Pose
```

若已经普通克隆：

```bash
git submodule update --init --recursive
```

子模块固定到 SMPLer-X 提交
`064baef0e4ab5277a3297691bc1d46ea5412586f`，保证适配器与上游接口一致。

### 2. 创建 Windows 推理环境并下载公开权重

在 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

脚本创建项目专用 `.venv`，安装 Python 3.9 对应的 PyTorch 1.12.1/CUDA 11.3、MMCV
1.7.1、上游依赖和 Gradio。若公开权重已经存在，可使用
`.\scripts\setup_windows.ps1 -SkipAssets`。

WSL2 备用方案：

```bash
bash scripts/setup_wsl.sh
conda activate occluded-sitting-smplx
```

WSL2 脚本使用 Python 3.8，并额外安装 OSMesa。两种方案都会下载以下公开文件：

- `models/checkpoints/smpler_x_s32.pth.tar`
- `models/detectors/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`
- `models/detectors/mmdet_faster_rcnn_r50_fpn_coco.py`

也可以只执行：

```bash
python scripts/download_assets.py
```

### 3. 添加受许可约束的中性人体模型

分别登录 [SMPL](https://smpl.is.tue.mpg.de/) 和
[SMPL-X](https://smpl-x.is.tue.mpg.de/) 官网，阅读并接受许可。姿势优先模式只需要两个
中性模型文件：

```text
models/body_models/
├── smpl/
│   └── SMPL_NEUTRAL.pkl
└── smplx/
    └── SMPLX_NEUTRAL.npz
```

上游代码会无条件加载训练/评测阶段才使用的性别模型和索引。项目启动时会在已忽略的
`models/runtime_body_models/` 中为这些**未使用项**生成中性模型别名和空索引占位符，不会
伪造或下载额外受限资产，也不会影响实际推理使用的中性 SMPL-X 层。

模型目录和运行时副本均被 `.gitignore` 排除。验证全部资产：

```bash
smplx-reconstruct check-assets
# 或
python scripts/check_assets.py
```

## 使用

### Web 界面

```bash
smplx-reconstruct serve
# 等价于 python app.py
```

访问 `http://127.0.0.1:7860`。上传图片后可选择：

- **自动检测最大人物**：适合人物轮廓较完整的单人图。
- **整张编辑结果为人物**：先用上传区裁剪到目标人物，适合多人、坐姿边界不完整或检测失败。

服务只监听本机且 `share=False`，图片不会发送给在线推理服务。为适配 6GB 显存，任务
串行执行。

### CLI

自动检测主人物：

```bash
smplx-reconstruct infer /path/to/person.jpg -o outputs
```

把整张图当作人物框：

```bash
smplx-reconstruct infer person.png -o outputs --full-image
```

显式指定源图片像素坐标中的 `x,y,width,height`：

```bash
smplx-reconstruct infer person.png -o outputs --bbox 120,40,760,980
```

默认使用 `--device cuda:0 --detector-device cpu`。每次调用会在输出根目录下创建唯一任务
目录，避免覆盖之前的结果。

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

## 输出格式

每个任务目录包含：

| 文件 | 内容 |
|---|---|
| `input.png` | 修正方向并转为 RGB 的输入 |
| `mesh.obj` | 米制相机坐标中的 SMPL-X 三角网格 |
| `smplx_params.npz` | 可重新生成姿态网格的 SMPL-X 兼容参数 |
| `metadata.json` | schema、模型版本、检测框、相机、耗时和显存 |
| `overlay.png` | 网格在原图上的投影 |
| `mesh_preview.png` | 正面、侧面、背面独立渲染 |
| `result.zip` | 上述文件的下载包 |

`smplx_params.npz` 字段均为 `float32`：

| 字段 | 形状 | 表示 |
|---|---:|---|
| `global_orient` | `(1, 3)` | 根节点轴角旋转 |
| `body_pose` | `(21, 3)` | 身体关节轴角旋转 |
| `left_hand_pose`, `right_hand_pose` | `(15, 3)` | 完整手部轴角旋转 |
| `jaw_pose`, `leye_pose`, `reye_pose` | `(1, 3)` | 固定为零的中性兼容字段 |
| `betas` | `(1, 10)` | 身体形状 |
| `expression` | `(1, 10)` | 固定为零；不保留预测表情 |
| `transl` | `(1, 3)` | 相机坐标平移，单位为米 |

## 坐姿与遮挡建议

- 尽量保留头、躯干、髋和可见腿部，避免裁剪掉座椅接触区域附近的身体线索。
- 检测框错误时先紧贴人物裁剪，再选择“整张编辑结果为人物”。
- 单目重建无法确定真实尺度或被完全遮挡肢体的唯一姿态；输出是模型依据训练先验给出的估计。
- 宽松衣物、极端透视、多人严重重叠、镜像和完全不可见的四肢可能产生明显误差。
- 本项目没有针对坐姿或遮挡重新训练，因此不宣称在这些场景中超过原始 SMPLer-X。

## 故障排查

### `CUDA is not available`

在 Windows 或 WSL2 中确认 `nvidia-smi` 可用，并检查 PyTorch：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### CUDA out of memory

关闭占用显存的软件，保持 `--detector-device cpu`，一次只提交一个任务。S32 是本项目唯一
后端；不会静默切换到质量或资源需求不同的模型。

### MMCV/MMDetection 导入失败

确认使用项目 `.venv`（或 WSL2 Conda 环境）与安装脚本指定的 PyTorch
1.12/CUDA 11.3/MMCV 1.7.1。不要升级到 MMCV 2.x 或 MMDetection 3.x。

### `pyrender` / OSMesa 错误

Windows 使用桌面 OpenGL，请更新 NVIDIA 驱动并从普通 PowerShell 启动服务；不要设置
`PYOPENGL_PLATFORM=osmesa`。WSL2/headless Linux 使用：

```bash
sudo apt-get install libosmesa6 libgl1-mesa-glx
export PYOPENGL_PLATFORM=osmesa
```

### 检测不到人物

在 Web 中裁剪到目标人物并选“整张编辑结果为人物”，或在 CLI 使用 `--full-image` / `--bbox`。

## 开发与验证

不需要权重或 GPU 的测试：

```bash
python -m pip install -e ".[dev]"
ruff check src tests scripts
pytest -m "not gpu"
```

使用自己的授权图片进行真实模型验收：

```bash
SMPLX_TEST_IMAGE=/path/to/person.jpg pytest -m gpu tests/test_gpu_integration.py
```

CI 只运行假后端测试，不下载受限资产。真实模型测试还会在 `metadata.json` 中记录
PyTorch 峰值显存，便于确认 RTX 2060 6GB 兼容性。

## 许可与引用

项目代码采用与 SMPLer-X 一致的 S-Lab License 1.0，仅限非商业用途。详见
[LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。使用本项目发表研究时，
请引用 SMPLer-X 与 SMPL-X：

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

## 发布到自己的 GitHub

上传前请确认 GitHub 仓库中不包含个人图片、`models/body_models/`、
`models/checkpoints/`、`outputs/` 或 `.venv/`。它们已经由 `.gitignore` 排除；公开仓库只
保留代码、配置、许可证、测试和公开模型下载脚本。建议先执行：

```powershell
git status --short
git ls-files | Select-String '\.(pth|pt|tar|pkl|npz|jpg|jpeg|png|zip)$'
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe -m pytest -m "not gpu"
```

本目录已经是 Git 仓库并包含固定子模块与 GitHub Actions。确认 Git 用户信息后提交，再将
`<USER>` 替换为你的 GitHub 用户名并添加远程：

```bash
git add .
git commit -m "Initial Single-Image-SMPLX-Pose release"
git remote add origin git@github.com:<USER>/Single-Image-SMPLX-Pose.git
git push -u origin main
```
