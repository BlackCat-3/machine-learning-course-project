$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
}
& '.\.venv\Scripts\python.exe' -m pip install --cache-dir .cache/pip -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Failed to install dependencies.' }
& '.\.venv\Scripts\python.exe' -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA GPU is not available'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'GPU environment verification failed.' }
