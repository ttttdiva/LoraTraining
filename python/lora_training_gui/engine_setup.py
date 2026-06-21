from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def run(args: list[str], cwd: Path | None = None) -> bool:
    print("[SETUP] " + " ".join(args), flush=True)
    try:
        subprocess.check_call(args, cwd=str(cwd) if cwd else None)
        return True
    except subprocess.CalledProcessError as error:
        print(f"[ERROR] command failed with exit code {error.returncode}", flush=True)
        return False


def venv_python(engine_root: Path) -> Path:
    return engine_root / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")


def ensure_venv(engine_root: Path) -> Path:
    python = venv_python(engine_root)
    if python.exists():
        print(f"[INFO] venv found: {python}", flush=True)
        return python

    print(f"[SETUP] creating venv: {engine_root / 'venv'}", flush=True)
    venv.create(engine_root / "venv", with_pip=True)
    if not python.exists():
        raise RuntimeError(f"venv python was not created: {python}")
    return python


def check_engine(engine_root: Path, engine_type: str) -> bool:
    if not engine_root.exists():
        print(f"[ERROR] engine root not found: {engine_root}", flush=True)
        return False

    required = ["requirements.txt"]
    if engine_type == "sd_scripts":
        required += ["train_network.py", "networks"]
    else:
        required += ["anima_train_network.py"]

    missing = [item for item in required if not (engine_root / item).exists()]
    if missing:
        print("[ERROR] engine files missing: " + ", ".join(missing), flush=True)
        return False

    print(f"[INFO] engine verified: {engine_root}", flush=True)
    return True


def nvidia_gpu_name() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except Exception:
        return ""


def pytorch_status(python: Path) -> str:
    code = r"""
import sys
try:
    import torch
    import torchvision
    if not torch.cuda.is_available():
        print("NO_CUDA")
        sys.exit(0)
    major, minor = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if major * 10 + minor >= 120 and "sm_120" not in arch_list:
        print("NEEDS_BLACKWELL")
    else:
        print("OK")
except ImportError:
    print("MISSING")
"""
    try:
        output = subprocess.check_output([str(python), "-c", code], text=True, stderr=subprocess.STDOUT)
        return output.strip().splitlines()[-1]
    except Exception:
        return "MISSING"


def install_pytorch(python: Path) -> bool:
    gpu_name = nvidia_gpu_name()
    status = pytorch_status(python)
    is_nvidia = bool(gpu_name)
    is_blackwell = "RTX 50" in gpu_name or status == "NEEDS_BLACKWELL"

    print(f"[INFO] GPU: {gpu_name or 'none'}", flush=True)
    print(f"[INFO] PyTorch status: {status}", flush=True)
    if status == "OK":
        return True

    if is_blackwell:
        print("[SETUP] RTX 50/Blackwell path: installing PyTorch nightly CUDA 13.0", flush=True)
        return run([str(python), "-m", "pip", "install", "--pre", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/nightly/cu130"])
    if is_nvidia:
        print("[SETUP] NVIDIA path: installing PyTorch CUDA 12.1", flush=True)
        return run([str(python), "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"])

    print("[SETUP] CPU/other GPU path: installing standard PyTorch", flush=True)
    return run([str(python), "-m", "pip", "install", "torch", "torchvision"])


def install_requirements(python: Path, engine_root: Path) -> bool:
    req = engine_root / "requirements.txt"
    if not req.exists():
        print(f"[INFO] no requirements.txt at {req}", flush=True)
        return True
    return run([str(python), "-m", "pip", "install", "-r", str(req)], cwd=engine_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--engine-type", default="anima_standalone")
    args = parser.parse_args()

    engine_root = Path(args.engine_root).expanduser().resolve()
    engine_type = str(args.engine_type or "anima_standalone")

    print("=" * 50, flush=True)
    print("LoraTraining Engine Setup", flush=True)
    print("=" * 50, flush=True)

    if not check_engine(engine_root, engine_type):
        return 1

    try:
        python = ensure_venv(engine_root)
    except Exception as error:
        print(f"[ERROR] {error}", flush=True)
        return 1

    if not run([str(python), "-m", "pip", "install", "--upgrade", "pip"]):
        return 1
    if not install_pytorch(python):
        return 1
    if not install_requirements(python, engine_root):
        return 1

    final_status = pytorch_status(python)
    if final_status not in {"OK", "NO_CUDA"}:
        print(f"[ERROR] final PyTorch verification failed: {final_status}", flush=True)
        return 1

    print("=" * 50, flush=True)
    print("Engine setup completed", flush=True)
    print("=" * 50, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
