"""HyperAI eğitimi öncesinde PyTorch CUDA ortamını doğrular."""

from __future__ import annotations

import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch


def main() -> None:
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'yok (CPU-only build)'}")
    print(f"CUDA kullanılabilir: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("GPU kullanılamıyor. requirements-cuda.txt ile CUDA PyTorch kurun.")
        raise SystemExit(1)
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GiB")
    sample = torch.randn((1024, 1024), device=device)
    result = sample @ sample.T
    torch.cuda.synchronize()
    print(f"CUDA tensör testi: PASS ({result.device})")


if __name__ == "__main__":
    main()
