[CmdletBinding()]
param(
    [switch]$SkipAssets
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

Set-Location $projectRoot
git submodule update --init --recursive

if (-not (Test-Path -LiteralPath $venvPython)) {
    py -3.9 -m venv $venvPath
}

& $venvPython -m pip install --upgrade "pip<25" "setuptools<70" wheel
& $venvPython -m pip install `
    "torch==1.12.1+cu113" `
    "torchvision==0.13.1+cu113" `
    "torchaudio==0.12.1" `
    --extra-index-url "https://download.pytorch.org/whl/cu113"
& $venvPython -m pip install "mmcv-full==1.7.1" `
    -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html"
& $venvPython -m pip install -e ".[web]"
& $venvPython -m pip install -r "third_party/SMPLer-X/requirements.txt"
& $venvPython -m pip install -v -e "third_party/SMPLer-X/main/transformer_utils"

# The upstream requirements are intentionally unpinned.  Restore the versions
# that match the S32 checkpoint and avoid optional boto3/urllib3 conflicts on
# Python 3.9 after installing them.
& $venvPython -m pip uninstall -y boto3 botocore s3transfer jmespath opencv-python timm
& $venvPython -m pip install --force-reinstall --no-deps `
    "numpy==1.23.5" `
    "opencv-python==4.10.0.84" `
    "opencv-python-headless==4.10.0.84" `
    "urllib3==2.2.3" `
    "timm==0.6.13" `
    "pyglet==1.5.27" `
    "pydantic==2.9.2"

if (-not $SkipAssets) {
    & $venvPython "scripts/download_assets.py"
}

& $venvPython -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
& $venvPython "scripts/check_assets.py"

Write-Host ""
Write-Host "Windows environment is ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
Write-Host "Start the Web UI with: smplx-reconstruct serve"
