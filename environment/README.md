# Benchmark Environment

This document records the primary environment used for the reported project experiments. To keep comparisons fair, the main benchmark models will be evaluated on the same machine and software environment.

## Hardware

- GPU: NVIDIA GeForce RTX 4090, 24 GB
- CPU: `<output from lscpu>`
- RAM: `<output from free -h>`

## Operating System

- OS: Ubuntu 22.04.5 LTS
- Kernel: `<output from uname -r>`
- NVIDIA driver: `<output from nvidia-smi>`
- Driver-supported CUDA version: 12.4
- System CUDA Toolkit (`nvcc`): `<version or not installed>`

## Python Environment

- Python: `<version>`
- pip: `<version>`
- PyTorch: `<version reported by torch.__version__>`
- torchvision: `<version reported by torchvision.__version__>`
- CUDA runtime bundled with PyTorch: `<value from torch.version.cuda>`
- cuDNN: `<value from torch.backends.cudnn.version()>`
- CUDA available to PyTorch: `<True or False>`

## Installation

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA 12.4 PyTorch build:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining project dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn pyyaml tqdm
```

## Verification

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected hardware result:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

## Benchmark Notes

- All main reported model comparisons will use this Ubuntu RTX 4090 environment.
- Mixed precision: `<enabled, disabled, or not decided>`
- Exact resolved packages are recorded in `requirements-lock.txt`.
- Timing methodology will be documented together with the final benchmark results.