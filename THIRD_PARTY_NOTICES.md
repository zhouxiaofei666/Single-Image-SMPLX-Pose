# Third-party notices

This repository integrates, but does not train, the following research software and models.
The root license does not replace any upstream terms.

## SMPLer-X

- Source: https://github.com/MotrixLab/SMPLer-X
- Pinned commit: `064baef0e4ab5277a3297691bc1d46ea5412586f`
- License: S-Lab License 1.0, non-commercial use only
- Copyright: S-Lab and the original contributors

The source is referenced as the Git submodule `third_party/SMPLer-X`. Its own `LICENSE`
file is authoritative.

## SMPL and SMPL-X

- SMPL: https://smpl.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- Body license information: https://smpl-x.is.tue.mpg.de/bodylicense.html

These model files are not included. Each user must register, accept the applicable terms,
download the files from the official websites, and use them within the granted license.

## MMDetection Faster R-CNN

- Project: https://github.com/open-mmlab/mmdetection
- Detector architecture/config: Faster R-CNN R50-FPN trained on COCO
- License: Apache License 2.0 for MMDetection code; checkpoint/dataset terms also apply

## Other dependencies

PyTorch, MMCV, MMPose, NumPy, Pillow, OpenCV, trimesh, pyrender, Gradio, and their
transitive packages remain subject to their respective licenses. See the package metadata
installed in the runtime environment for exact versions and notices.
