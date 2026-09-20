# Assignment 1 Benchmark Environment

This environment produced the six reported Linear/MLP runs for the 23 September
milestone.

## Hardware and operating system

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB.
- CPU: 13th Gen Intel Core i7-13620H.
- NVIDIA driver: 596.36.
- Windows build: `10.0.26200.9457`.
- Shell: PowerShell.

## Python environment

- Conda environment: `uav`.
- Python: 3.11.15.
- pip: 26.1.1.
- PyTorch: 2.5.1+cu121.
- torchvision: 0.20.1+cu121.
- CUDA runtime bundled with PyTorch: 12.1.
- cuDNN reported by PyTorch: 90100.

Remaining direct dependencies are pinned in `requirements.txt`.

## Verification

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Expected benchmark output:

```text
2.5.1+cu121
0.20.1+cu121
True
12.1
NVIDIA GeForce RTX 4050 Laptop GPU
```

## Benchmark notes

- Float32 precision; automatic mixed precision disabled.
- Batch size 128 and DataLoader workers 0.
- Deterministic PyTorch operations enabled; cuDNN benchmarking disabled.
- Forward-only inference timing with five warm-up batches and three repetitions.
- CUDA synchronization around every timed forward pass.
- Training time can be affected by normal laptop activity; predictive metrics
  are deterministic for a fixed seed and configuration.
