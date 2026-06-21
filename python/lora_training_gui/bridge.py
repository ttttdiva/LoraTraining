from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWN_TOOL_ROOT = Path("D:/tool/lora_trainer")
APP_DATA = PROJECT_ROOT / "data"
SETTINGS_PATH = APP_DATA / "settings.json"
DEFAULT_TAGGER_MODEL_DIR = Path("D:/AI/models/Hot/image/Tagger")
DEFAULT_ENGINE_ROOT = KNOWN_TOOL_ROOT / "Anima-Standalone-Trainer"
DEFAULT_SD_SCRIPTS_ROOT = KNOWN_TOOL_ROOT / "sd-scripts"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "trash"}


def success(job: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "ok",
        "job": job,
        "data": data,
        "warnings": warnings or [],
        "errors": [],
    }


def failure(job: str, errors: list[str], data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "error",
        "job": job,
        "data": data or {},
        "warnings": [],
        "errors": errors,
    }


def run_command(args: list[str], timeout: float = 5.0) -> tuple[bool, str, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as error:
        return False, "", str(error)
    except subprocess.TimeoutExpired as error:
        return False, error.stdout or "", f"timed out after {timeout}s"

    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()


def detect_tool(name: str, version_args: list[str]) -> dict[str, Any]:
    executable = shutil.which(version_args[0])
    if not executable:
        return {
            "name": name,
            "available": False,
            "error": f"{version_args[0]} not found in PATH",
        }

    command_args = [executable, *version_args[1:]]
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        command_args = ["cmd", "/d", "/c", executable, *version_args[1:]]

    ok, stdout, stderr = run_command(command_args)
    return {
        "name": name,
        "available": ok,
        "version": (stdout or stderr).splitlines()[0] if (stdout or stderr) else "",
        "path": executable,
        "error": None if ok else stderr or stdout or "version command failed",
    }


def parse_int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except ValueError:
        return None


def detect_gpus() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {
            "available": False,
            "gpus": [],
            "error": "nvidia-smi not found in PATH",
        }

    query = "index,name,memory.total,driver_version,temperature.gpu,utilization.gpu"
    ok, stdout, stderr = run_command(
        [
            smi,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        timeout=5.0,
    )
    if not ok:
        return {
            "available": False,
            "gpus": [],
            "error": stderr or stdout or "nvidia-smi failed",
        }

    gpus: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue

        index = parse_int(parts[0])
        if index is None:
            continue

        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "memoryTotalMiB": parse_int(parts[2]),
                "driverVersion": parts[3],
                "temperatureC": parse_int(parts[4]),
                "utilizationPct": parse_int(parts[5]),
            }
        )

    return {
        "available": len(gpus) > 0,
        "gpus": gpus,
        "error": None if gpus else "nvidia-smi returned no GPUs",
    }


def detect_engines() -> list[dict[str, Any]]:
    candidates = [
        {
            "id": "anima-standalone",
            "name": "Anima Standalone Trainer",
            "type": "anima_standalone",
            "root": KNOWN_TOOL_ROOT / "Anima-Standalone-Trainer",
            "required": ["anima_train.py", "training-ui/architectures.json"],
            "notes": [
                "First execution adapter candidate.",
                "Includes job TOML, multi-GPU, and sample generation references.",
            ],
        },
        {
            "id": "sd-scripts",
            "name": "kohya-ss sd-scripts",
            "type": "sd_scripts",
            "root": KNOWN_TOOL_ROOT / "sd-scripts",
            "required": ["train_network.py", "sdxl_train_network.py", "library", "networks", "requirements.txt"],
            "notes": [
                "Official sd-scripts clone for SD1/SD2/SDXL LoRA execution.",
                "Used directly as a selectable training engine.",
            ],
        },
        {
            "id": "kohya-param-gui",
            "name": "Kohya lora param GUI",
            "type": "sd_scripts_reference",
            "root": KNOWN_TOOL_ROOT / "Kohya_lora_param_gui",
            "required": ["Kohya_lora_trainer/TrainParams.cs", "Kohya_lora_trainer/MyUtils.cs"],
            "notes": [
                "Source reference for the parameter registry.",
                "Used as a settings reference rather than a direct execution engine.",
            ],
        },
        {
            "id": "anima-lora-factory",
            "name": "Anima LoRA Factory",
            "type": "workflow_reference",
            "root": KNOWN_TOOL_ROOT / "Anima-LoRA-Factory",
            "required": ["README.md"],
            "notes": [
                "Reference for onboarding, preprocessing, and tag editing workflow.",
                "Fixed-port browser GUI behavior will not be copied.",
            ],
        },
    ]

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        root = Path(candidate["root"])
        missing = [rel for rel in candidate["required"] if not (root / rel).exists()]
        notes = list(candidate["notes"])
        if missing:
            notes.append("missing: " + ", ".join(missing))

        results.append(
            {
                "id": candidate["id"],
                "name": candidate["name"],
                "type": candidate["type"],
                "root": str(root),
                "available": root.exists() and not missing,
                "notes": notes,
            }
        )

    return results


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(read_text(path))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or f"job_{int(time.time())}"


def known_anima_venv() -> Path:
    return DEFAULT_ENGINE_ROOT / "venv"


def default_engines() -> list[dict[str, str]]:
    sd_scripts_venv = DEFAULT_SD_SCRIPTS_ROOT / "venv"
    shared_venv = sd_scripts_venv if sd_scripts_venv.exists() else known_anima_venv()
    return [
        {
            "id": "anima-standalone",
            "name": "Anima Standalone Trainer",
            "type": "anima_standalone",
            "root": str(DEFAULT_ENGINE_ROOT),
            "venv": str(known_anima_venv()),
        },
        {
            "id": "sd-scripts",
            "name": "kohya-ss sd-scripts",
            "type": "sd_scripts",
            "root": str(DEFAULT_SD_SCRIPTS_ROOT),
            "venv": str(shared_venv),
        },
    ]


def default_settings() -> dict[str, Any]:
    jobs_root = APP_DATA / "jobs"
    return {
        "schemaVersion": 1,
        "jobsRoot": str(jobs_root),
        "defaultEngineId": "anima-standalone",
        "taggerModelDir": str(DEFAULT_TAGGER_MODEL_DIR),
        "engines": default_engines(),
        "defaults": {
            "architecture": "anima",
            "captionExtension": ".txt",
            "mixedPrecision": "bf16",
            "gpuIds": "0",
            "multiGpuMode": "single",
        },
    }


def merge_engines(stored_engines: Any) -> list[dict[str, Any]]:
    defaults = default_engines()
    if not isinstance(stored_engines, list):
        return defaults

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    default_by_id = {engine["id"]: engine for engine in defaults}
    for engine in stored_engines:
        if not isinstance(engine, dict) or not engine.get("id"):
            continue
        engine_id = str(engine["id"])
        base = default_by_id.get(engine_id, {})
        item = {**base, **engine}
        merged.append(item)
        seen.add(engine_id)

    for engine in defaults:
        if engine["id"] not in seen:
            merged.append(engine)
    return merged


def load_settings_dict() -> dict[str, Any]:
    settings = default_settings()
    stored = read_json(SETTINGS_PATH, {})
    if isinstance(stored, dict):
        settings.update(stored)
        if isinstance(stored.get("defaults"), dict):
            settings["defaults"].update(stored["defaults"])
        settings["engines"] = merge_engines(stored.get("engines"))
    return settings


def save_settings_dict(settings: dict[str, Any]) -> dict[str, Any]:
    merged = default_settings()
    merged.update(settings)
    if isinstance(settings.get("defaults"), dict):
        merged["defaults"].update(settings["defaults"])
    merged["engines"] = merge_engines(settings.get("engines"))
    write_json(SETTINGS_PATH, merged)
    return merged


def settings_get(_: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings_dict()
    Path(settings["jobsRoot"]).mkdir(parents=True, exist_ok=True)
    Path(settings["taggerModelDir"]).mkdir(parents=True, exist_ok=True)
    return success("settings_get", settings)


def settings_save(payload: dict[str, Any]) -> dict[str, Any]:
    settings = save_settings_dict(payload)
    Path(settings["jobsRoot"]).mkdir(parents=True, exist_ok=True)
    Path(settings["taggerModelDir"]).mkdir(parents=True, exist_ok=True)
    return success("settings_save", settings)


def get_engine(settings: dict[str, Any], engine_id: str | None = None) -> dict[str, Any]:
    target_id = engine_id or settings.get("defaultEngineId") or "anima-standalone"
    engines = settings.get("engines") or []
    for engine in engines:
        if engine.get("id") == target_id:
            return engine
    return default_settings()["engines"][0]


def jobs_root(settings: dict[str, Any] | None = None) -> Path:
    current = settings or load_settings_dict()
    root = Path(str(current.get("jobsRoot") or APP_DATA / "jobs")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_name: str, settings: dict[str, Any] | None = None) -> Path:
    return jobs_root(settings) / sanitize_name(job_name)


def job_json_path(job_name: str, settings: dict[str, Any] | None = None) -> Path:
    return job_dir(job_name, settings) / "job.json"


def default_job(name: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    current_settings = settings or load_settings_dict()
    defaults = current_settings.get("defaults", {})
    safe_name = sanitize_name(name)
    engine = get_engine(current_settings)
    return {
        "schemaVersion": 1,
        "name": safe_name,
        "displayName": name.strip() or safe_name,
        "engineId": engine.get("id", "anima-standalone"),
        "architecture": defaults.get("architecture", "anima"),
        "modelPaths": {
            "baseModelPath": "",
            "ditPath": "",
            "qwen3Path": "",
            "vaePath": "",
        },
        "dataset": {
            "imageDir": "",
            "captionExtension": defaults.get("captionExtension", ".txt"),
            "resolution": [1536, 1536],
            "batchSize": 1,
            "numRepeats": 1,
            "keepTokens": 1,
            "shuffleCaption": False,
            "captionDropoutRate": 0.05,
            "captionTagDropoutRate": 0.0,
            "enableBucket": True,
            "bucketNoUpscale": True,
            "minBucketReso": 512,
            "maxBucketReso": 1536,
            "bucketResoSteps": 64,
        },
        "training": {
            "outputName": safe_name,
            "maxTrainSteps": 0,
            "maxTrainEpochs": 15,
            "saveEveryNEpochs": 1,
            "saveEveryNSteps": 0,
            "learningRate": 0.0001,
            "unetLr": 0.0001,
            "textEncoderLr": 0.00005,
            "optimizerType": "AdamW8bit",
            "optimizerArgs": ["weight_decay=0.01"],
            "lrScheduler": "cosine",
            "lrWarmupSteps": 0,
            "mixedPrecision": defaults.get("mixedPrecision", "bf16"),
            "gradientCheckpointing": True,
            "gradientAccumulationSteps": 1,
            "maxDataLoaderNWorkers": 4,
            "persistentDataLoaderWorkers": True,
            "seed": 42,
            "cacheLatentsToDisk": True,
            "cacheTextEncoderOutputsToDisk": True,
            "sampleEveryNEpochs": 1,
            "sampleEveryNSteps": 0,
            "logWith": "tensorboard",
            "sdpa": True,
            "xformers": False,
            "clipSkip": 0,
        },
        "network": {
            "module": "networks.lora_anima",
            "dim": 16,
            "alpha": 16,
            "args": [],
            "trainUnetOnly": True,
            "resume": "",
            "networkWeights": "",
        },
        "anima": {
            "timestepSampleMethod": "logit_normal",
            "discreteFlowShift": 3.0,
            "weightingScheme": "logit_normal",
        },
        "sdScripts": {
            "extraArgs": "",
            "v2": False,
            "vParameterization": False,
        },
        "gpu": {
            "ids": defaults.get("gpuIds", "0"),
            "mode": defaults.get("multiGpuMode", "single"),
            "tpBackend": "gloo" if os.name == "nt" else "nccl",
        },
        "sample": {
            "prompts": "",
            "steps": 30,
            "sampler": "euler_a",
            "scale": 7.0,
            "keepLoaded": False,
            "networkMul": 1.0,
            "gpuIds": defaults.get("gpuIds", "0"),
            "multiGpuMode": "parallel_cfg",
            "extraArgs": "",
        },
        "wandb": {
            "mode": "disabled",
            "project": "",
            "entity": "",
            "runName": "",
            "tags": "",
        },
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def normalize_job(job: dict[str, Any]) -> dict[str, Any]:
    base = default_job(str(job.get("name") or "job"))
    for key, value in job.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    base["name"] = sanitize_name(str(base.get("name") or base.get("displayName") or "job"))
    base["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return base


def save_job_dict(job: dict[str, Any], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    current_settings = settings or load_settings_dict()
    normalized = normalize_job(job)
    path = job_json_path(normalized["name"], current_settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, normalized)
    ensure_job_text_files(normalized, current_settings)
    return normalized


def load_job_dict(job_name: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    path = job_json_path(job_name, settings)
    if not path.exists():
        raise FileNotFoundError(f"job not found: {job_name}")
    return normalize_job(read_json(path, {}))


def ensure_job_text_files(job: dict[str, Any], settings: dict[str, Any] | None = None) -> None:
    root = job_dir(job["name"], settings)
    root.mkdir(parents=True, exist_ok=True)
    prompts_path = root / "sample_prompts.txt"
    if not prompts_path.exists():
        write_text(prompts_path, str(job.get("sample", {}).get("prompts") or ""))


def jobs_list(_: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings_dict()
    root = jobs_root(settings)
    jobs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = normalize_job(read_json(path, {}))
            jobs.append(
                {
                    "name": job["name"],
                    "displayName": job.get("displayName", job["name"]),
                    "engineId": job.get("engineId"),
                    "architecture": job.get("architecture"),
                    "datasetImageDir": job.get("dataset", {}).get("imageDir", ""),
                    "updatedAt": job.get("updatedAt", ""),
                    "path": str(path.parent),
                }
            )
        except Exception:
            continue
    return success("jobs_list", {"jobsRoot": str(root), "jobs": jobs})


def job_create(payload: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings_dict()
    name = sanitize_name(str(payload.get("name") or "new_lora"))
    job = default_job(name, settings)
    if isinstance(payload.get("job"), dict):
        job.update(payload["job"])
    saved = save_job_dict(job, settings)
    return success("job_create", {"job": saved, "path": str(job_dir(saved["name"], settings))})


def job_get(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        return failure("job_get", ["name is required"])
    settings = load_settings_dict()
    job = load_job_dict(name, settings)
    return success("job_get", {"job": job, "path": str(job_dir(job["name"], settings))})


def job_save(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("job"), dict):
        return failure("job_save", ["job object is required"])
    settings = load_settings_dict()
    saved = save_job_dict(payload["job"], settings)
    return success("job_save", {"job": saved, "path": str(job_dir(saved["name"], settings))})


def job_clone(payload: dict[str, Any]) -> dict[str, Any]:
    source = str(payload.get("source") or "").strip()
    target = sanitize_name(str(payload.get("target") or f"{source}_copy"))
    if not source:
        return failure("job_clone", ["source is required"])
    settings = load_settings_dict()
    job = load_job_dict(source, settings)
    job["name"] = target
    job["displayName"] = target
    job["training"]["outputName"] = target
    saved = save_job_dict(job, settings)
    return success("job_clone", {"job": saved, "path": str(job_dir(saved["name"], settings))})


def job_delete(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        return failure("job_delete", ["name is required"])
    target = job_dir(name)
    if target.exists():
        shutil.rmtree(target)
    return success("job_delete", {"deleted": name})


def toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).lower()
    if isinstance(value, list):
        return "[ " + ", ".join(toml_scalar(item) for item in value) + " ]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def toml_section(name: str, values: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    for key, value in values.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} = {toml_scalar(value)}")
    return "\n".join(lines)


def write_dataset_toml(job: dict[str, Any], path: Path) -> None:
    dataset = job["dataset"]
    lines = [
        "[general]",
        f"enable_bucket = {toml_scalar(dataset.get('enableBucket', True))}",
        f"bucket_no_upscale = {toml_scalar(dataset.get('bucketNoUpscale', True))}",
        f"min_bucket_reso = {toml_scalar(dataset.get('minBucketReso', 512))}",
        f"max_bucket_reso = {toml_scalar(dataset.get('maxBucketReso', 1536))}",
        f"bucket_reso_steps = {toml_scalar(dataset.get('bucketResoSteps', 64))}",
        "",
        "[[datasets]]",
        f"resolution = {toml_scalar(dataset.get('resolution', [1536, 1536]))}",
        f"batch_size = {toml_scalar(dataset.get('batchSize', 1))}",
        "",
        "  [[datasets.subsets]]",
        f"  image_dir = {toml_scalar(dataset.get('imageDir', ''))}",
        f"  caption_extension = {toml_scalar(dataset.get('captionExtension', '.txt'))}",
        f"  num_repeats = {toml_scalar(dataset.get('numRepeats', 1))}",
        f"  keep_tokens = {toml_scalar(dataset.get('keepTokens', 1))}",
        "  flip_aug = false",
        '  caption_prefix = ""',
        f"  shuffle_caption = {toml_scalar(dataset.get('shuffleCaption', False))}",
        f"  caption_tag_dropout_rate = {toml_scalar(dataset.get('captionTagDropoutRate', 0.0))}",
        f"  caption_dropout_rate = {toml_scalar(dataset.get('captionDropoutRate', 0.05))}",
    ]
    write_text(path, "\n".join(lines) + "\n")


def merged_config_for_job(job: dict[str, Any], root: Path, dataset_path: Path, prompts_path: Path) -> dict[str, dict[str, Any]]:
    training = job["training"]
    network = job["network"]
    anima = job["anima"]
    model = job["modelPaths"]
    output_dir = root / "output"
    logs_dir = root / "logs"
    config: dict[str, dict[str, Any]] = {
        "model_arguments": {
            "dit_path": model.get("ditPath", ""),
            "qwen3_path": model.get("qwen3Path", ""),
            "vae_path": model.get("vaePath", ""),
        },
        "dataset_arguments": {
            "dataset_config": str(dataset_path),
            "cache_latents_to_disk": training.get("cacheLatentsToDisk", True),
            "cache_text_encoder_outputs_to_disk": training.get("cacheTextEncoderOutputsToDisk", True),
        },
        "training_arguments": {
            "output_dir": str(output_dir),
            "output_name": training.get("outputName", job["name"]),
            "save_model_as": "safetensors",
            "max_train_epochs": training.get("maxTrainEpochs", 15),
            "save_every_n_epochs": training.get("saveEveryNEpochs", 1),
            "logging_dir": str(logs_dir),
            "log_with": training.get("logWith", "tensorboard"),
            "learning_rate": training.get("learningRate", 0.0001),
            "text_encoder_lr": training.get("textEncoderLr", 0.00005),
            "optimizer_type": training.get("optimizerType", "AdamW8bit"),
            "optimizer_args": training.get("optimizerArgs", []),
            "lr_scheduler": training.get("lrScheduler", "cosine"),
            "lr_warmup_steps": training.get("lrWarmupSteps", 0),
            "mixed_precision": training.get("mixedPrecision", "bf16"),
            "gradient_checkpointing": training.get("gradientCheckpointing", True),
            "gradient_accumulation_steps": training.get("gradientAccumulationSteps", 1),
            "max_data_loader_n_workers": training.get("maxDataLoaderNWorkers", 4),
            "persistent_data_loader_workers": training.get("persistentDataLoaderWorkers", True),
            "seed": training.get("seed", 42),
            "save_state": True,
            "save_last_n_epochs_state": 1,
        },
        "network_arguments": {
            "network_module": network.get("module", "networks.lora_anima"),
            "network_dim": network.get("dim", 16),
            "network_alpha": network.get("alpha", 16),
            "network_train_unet_only": network.get("trainUnetOnly", True),
        },
        "anima_arguments": {
            "timestep_sample_method": anima.get("timestepSampleMethod", "logit_normal"),
            "discrete_flow_shift": anima.get("discreteFlowShift", 3.0),
            "weighting_scheme": anima.get("weightingScheme", "logit_normal"),
        },
    }
    if network.get("resume"):
        config["training_arguments"]["resume"] = network["resume"]
    if network.get("networkWeights"):
        config["network_arguments"]["network_weights"] = network["networkWeights"]
    if prompts_path.exists() and read_text(prompts_path).strip():
        config["sample_arguments"] = {
            "sample_prompts": str(prompts_path),
            "sample_every_n_epochs": training.get("sampleEveryNEpochs", 1),
        }
    return config


def write_toml_config(config: dict[str, dict[str, Any]], path: Path) -> None:
    sections = [toml_section(name, values) for name, values in config.items()]
    write_text(path, "\n\n".join(sections) + "\n")


def build_job_files(job: dict[str, Any], settings: dict[str, Any] | None = None) -> dict[str, str]:
    current_settings = settings or load_settings_dict()
    root = job_dir(job["name"], current_settings)
    root.mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)

    dataset_path = root / "dataset.toml"
    prompts_path = root / "sample_prompts.txt"
    merged_config_path = root / "_merged_config.toml"

    write_dataset_toml(job, dataset_path)
    write_text(prompts_path, str(job.get("sample", {}).get("prompts") or ""))
    config = merged_config_for_job(job, root, dataset_path, prompts_path)
    write_toml_config(config, merged_config_path)
    save_job_dict(job, current_settings)

    return {
        "root": str(root),
        "datasetToml": str(dataset_path),
        "samplePrompts": str(prompts_path),
        "mergedConfig": str(merged_config_path),
        "outputDir": str(root / "output"),
        "logsDir": str(root / "logs"),
        "samplesDir": str(root / "samples"),
    }


def job_build_files(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        return failure("job_build_files", ["name is required"])
    settings = load_settings_dict()
    job = load_job_dict(name, settings)
    files = build_job_files(job, settings)
    return success("job_build_files", {"job": job, "files": files})


def split_gpu_ids(gpu_ids: str) -> list[str]:
    return [part.strip() for part in str(gpu_ids or "").split(",") if part.strip()]


def engine_python(engine: dict[str, Any]) -> str:
    venv = Path(str(engine.get("venv") or Path(engine.get("root", "")) / "venv"))
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if candidate.exists():
        return str(candidate)
    return sys.executable


def build_accelerate_flags(job: dict[str, Any]) -> list[str]:
    gpu = job.get("gpu", {})
    ids = split_gpu_ids(gpu.get("ids", ""))
    mode = gpu.get("mode", "single")
    mixed = job.get("training", {}).get("mixedPrecision", "bf16")
    flags = ["--num_cpu_threads_per_process", "1"]
    if len(ids) <= 1 or mode == "single":
        flags += ["--mixed_precision", mixed]
    elif mode == "fsdp2":
        flags += ["--use_fsdp", "--fsdp_version", "2", "--num_processes", str(len(ids)), "--mixed_precision", mixed]
    elif mode == "fsdp":
        flags += ["--use_fsdp", "--fsdp_version", "1", "--num_processes", str(len(ids)), "--mixed_precision", mixed]
    elif mode == "deepspeed":
        flags += ["--use_deepspeed", "--zero_stage", "2", "--num_processes", str(len(ids)), "--mixed_precision", mixed]
    else:
        flags += ["--multi_gpu", "--num_processes", str(len(ids)), "--mixed_precision", mixed]
    return flags


def process_env_for_job(job: dict[str, Any]) -> dict[str, str]:
    ids = split_gpu_ids(job.get("gpu", {}).get("ids", ""))
    env = {
        "PYTHONIOENCODING": "utf-8",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    if os.name == "nt" and len(ids) > 1:
        env["USE_LIBUV"] = "0"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29500"
    wandb = job.get("wandb", {})
    if wandb.get("mode") and wandb.get("mode") != "disabled":
        env["WANDB_MODE"] = wandb.get("mode")
        if wandb.get("project"):
            env["WANDB_PROJECT"] = wandb["project"]
        if wandb.get("entity"):
            env["WANDB_ENTITY"] = wandb["entity"]
    return env


def looks_like_local_path(path_value: str) -> bool:
    if not path_value:
        return False
    if re.match(r"^[A-Za-z]:[\\/]", path_value):
        return True
    if path_value.startswith(("/", "\\", "~", ".")):
        return True
    return Path(path_value).suffix.lower() in {".safetensors", ".ckpt", ".pt", ".bin"}


def model_path_errors(job: dict[str, Any], engine_type: str = "anima_standalone") -> list[str]:
    errors: list[str] = []
    model_paths = job.get("modelPaths", {})
    if engine_type == "sd_scripts":
        base_model = str(model_paths.get("baseModelPath") or "").strip()
        if not base_model:
            errors.append("model path is empty: baseModelPath")
        elif looks_like_local_path(base_model) and not Path(base_model).expanduser().exists():
            errors.append(f"model path not found: baseModelPath={base_model}")
        vae_path = str(model_paths.get("vaePath") or "").strip()
        if vae_path and looks_like_local_path(vae_path) and not Path(vae_path).expanduser().exists():
            errors.append(f"model path not found: vaePath={vae_path}")
        return errors

    for key, path_value in job.get("modelPaths", {}).items():
        if key == "baseModelPath":
            continue
        if not str(path_value).strip():
            errors.append(f"model path is empty: {key}")
        elif not Path(str(path_value)).exists():
            errors.append(f"model path not found: {key}={path_value}")
    return errors


def parse_arg_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

    args: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--") and "=" in line:
            args.append(line)
        else:
            args.extend(shlex.split(line, comments=True, posix=True))
    return args


def add_flag(args: list[str], enabled: Any, flag: str) -> None:
    if bool(enabled):
        args.append(flag)


def add_option(args: list[str], key: str, value: Any, skip_empty: bool = True) -> None:
    if skip_empty and (value is None or value == ""):
        return
    args.extend([key, str(value)])


def sd_scripts_train_script(job: dict[str, Any]) -> str:
    architecture = str(job.get("architecture") or "sd15").lower()
    if architecture == "sdxl":
        return "sdxl_train_network.py"
    if architecture in {"sd15", "sd1", "sd2"}:
        return "train_network.py"
    return "train_network.py"


def sd_scripts_gen_script(job: dict[str, Any]) -> str:
    architecture = str(job.get("architecture") or "sd15").lower()
    return "sdxl_gen_img.py" if architecture == "sdxl" else "gen_img.py"


def caption_randomization_enabled(job: dict[str, Any]) -> bool:
    dataset = job.get("dataset", {})
    return bool(dataset.get("shuffleCaption")) or float(dataset.get("captionDropoutRate") or 0) > 0 or float(dataset.get("captionTagDropoutRate") or 0) > 0


def sd_scripts_option_errors(job: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    architecture = str(job.get("architecture") or "sd15").lower()
    training = job.get("training", {})
    if architecture == "sdxl" and training.get("cacheTextEncoderOutputsToDisk") and caption_randomization_enabled(job):
        errors.append("SDXL cacheTextEncoderOutputsToDisk cannot be used with shuffle/caption dropout/tag dropout")
    return errors


def build_sd_scripts_args(job: dict[str, Any], files: dict[str, str], script: Path) -> list[str]:
    training = job.get("training", {})
    network = job.get("network", {})
    model = job.get("modelPaths", {})
    sd_scripts = job.get("sdScripts", {})
    architecture = str(job.get("architecture") or "sd15").lower()
    network_module = str(network.get("module") or "networks.lora")
    if network_module == "networks.lora_anima":
        network_module = "networks.lora"

    args = [
        "-m",
        "accelerate.commands.launch",
        *build_accelerate_flags(job),
        str(script),
        "--pretrained_model_name_or_path",
        str(model.get("baseModelPath") or ""),
        "--dataset_config",
        files["datasetToml"],
        "--output_dir",
        files["outputDir"],
        "--output_name",
        str(training.get("outputName") or job["name"]),
        "--save_model_as",
        "safetensors",
        "--network_module",
        network_module,
        "--network_dim",
        str(network.get("dim", 16)),
        "--network_alpha",
        str(network.get("alpha", 1)),
        "--learning_rate",
        str(training.get("learningRate", 0.0001)),
        "--optimizer_type",
        str(training.get("optimizerType") or "AdamW8bit"),
        "--lr_scheduler",
        str(training.get("lrScheduler") or "constant"),
        "--lr_warmup_steps",
        str(training.get("lrWarmupSteps", 0)),
        "--mixed_precision",
        str(training.get("mixedPrecision") or "fp16"),
        "--gradient_accumulation_steps",
        str(training.get("gradientAccumulationSteps", 1)),
        "--max_data_loader_n_workers",
        str(training.get("maxDataLoaderNWorkers", 4)),
        "--seed",
        str(training.get("seed", 42)),
        "--logging_dir",
        files["logsDir"],
    ]

    add_option(args, "--text_encoder_lr", training.get("textEncoderLr"))
    add_option(args, "--unet_lr", training.get("unetLr"))
    add_option(args, "--save_every_n_epochs", training.get("saveEveryNEpochs"))
    if int(training.get("saveEveryNSteps") or 0) > 0:
        add_option(args, "--save_every_n_steps", training.get("saveEveryNSteps"))
    if int(training.get("maxTrainEpochs") or 0) > 0:
        add_option(args, "--max_train_epochs", training.get("maxTrainEpochs"))
    elif int(training.get("maxTrainSteps") or 0) > 0:
        add_option(args, "--max_train_steps", training.get("maxTrainSteps"))
    if training.get("logWith") in {"tensorboard", "wandb", "all"}:
        add_option(args, "--log_with", training.get("logWith"))
    if model.get("vaePath"):
        add_option(args, "--vae", model.get("vaePath"))
    if int(training.get("clipSkip") or 0) > 0 and architecture != "sdxl":
        add_option(args, "--clip_skip", training.get("clipSkip"))
    if int(training.get("sampleEveryNEpochs") or 0) > 0 and read_text(Path(files["samplePrompts"])).strip():
        add_option(args, "--sample_every_n_epochs", training.get("sampleEveryNEpochs"))
        add_option(args, "--sample_prompts", files["samplePrompts"])
    if int(training.get("sampleEveryNSteps") or 0) > 0 and read_text(Path(files["samplePrompts"])).strip():
        add_option(args, "--sample_every_n_steps", training.get("sampleEveryNSteps"))
        add_option(args, "--sample_prompts", files["samplePrompts"])

    optimizer_args = parse_arg_list(training.get("optimizerArgs"))
    if optimizer_args:
        args.extend(["--optimizer_args", *optimizer_args])
    network_args = parse_arg_list(network.get("args"))
    if network_args:
        args.extend(["--network_args", *network_args])

    add_flag(args, training.get("gradientCheckpointing", True), "--gradient_checkpointing")
    add_flag(args, training.get("sdpa", True), "--sdpa")
    add_flag(args, training.get("xformers", False), "--xformers")
    add_flag(args, training.get("cacheLatentsToDisk", True), "--cache_latents_to_disk")
    add_flag(args, network.get("trainUnetOnly", False), "--network_train_unet_only")
    add_flag(args, sd_scripts.get("v2", architecture == "sd2"), "--v2")
    add_flag(args, sd_scripts.get("vParameterization", False), "--v_parameterization")

    if architecture == "sdxl" and not caption_randomization_enabled(job):
        add_flag(args, training.get("cacheTextEncoderOutputsToDisk", False), "--cache_text_encoder_outputs_to_disk")

    if network.get("resume"):
        add_option(args, "--resume", network.get("resume"))
    if network.get("networkWeights"):
        add_option(args, "--network_weights", network.get("networkWeights"))

    args.extend(parse_arg_list(sd_scripts.get("extraArgs")))
    return args


def build_sd_scripts_sample_args(job: dict[str, Any], files: dict[str, str], script: Path, network_weights: str) -> list[str]:
    training = job.get("training", {})
    sample = job.get("sample", {})
    network = job.get("network", {})
    model = job.get("modelPaths", {})
    dataset = job.get("dataset", {})
    width, height = (dataset.get("resolution") or [512, 512])[:2]
    network_module = str(network.get("module") or "networks.lora")
    if network_module == "networks.lora_anima":
        network_module = "networks.lora"
    args = [
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        "1",
        str(script),
        "--ckpt",
        str(model.get("baseModelPath") or ""),
        "--from_file",
        files["samplePrompts"],
        "--outdir",
        files["samplesDir"],
        "--W",
        str(width),
        "--H",
        str(height),
        "--steps",
        str(sample.get("steps") or 30),
        "--sampler",
        str(sample.get("sampler") or "euler_a"),
        "--scale",
        str(sample.get("scale") or 7.0),
        "--seed",
        str(training.get("seed", 42)),
    ]
    precision = str(training.get("mixedPrecision") or "").lower()
    if precision == "bf16":
        args.append("--bf16")
    elif precision == "fp16":
        args.append("--fp16")
    if model.get("vaePath"):
        add_option(args, "--vae", model.get("vaePath"))
    if training.get("sdpa", True):
        args.append("--sdpa")
    if training.get("xformers", False):
        args.append("--xformers")
    if int(training.get("clipSkip") or 0) > 0:
        add_option(args, "--clip_skip", training.get("clipSkip"))
    if network_weights:
        args.extend(["--network_module", network_module])
        args.extend(["--network_weights", network_weights])
        args.extend(["--network_mul", str(sample.get("networkMul") or 1.0)])
    network_args = parse_arg_list(network.get("args"))
    if network_args:
        args.extend(["--network_args", *network_args])
    args.extend(parse_arg_list(sample.get("extraArgs")))
    return args


def quote_arg(arg: str) -> str:
    if re.search(r"\s|\"", arg):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def make_plan(process_id: str, kind: str, cwd: Path, command: str, args: list[str], env: dict[str, str], files: dict[str, str] | None = None, url: str | None = None) -> dict[str, Any]:
    return {
        "id": process_id,
        "kind": kind,
        "cwd": str(cwd),
        "command": command,
        "args": args,
        "env": env,
        "displayCommand": " ".join([quote_arg(command), *[quote_arg(str(arg)) for arg in args]]),
        "files": files or {},
        "url": url,
    }


def train_launch_plan(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        return failure("train_launch_plan", ["name is required"])
    settings = load_settings_dict()
    job = load_job_dict(name, settings)
    engine = get_engine(settings, job.get("engineId"))
    engine_type = str(engine.get("type") or "anima_standalone")
    engine_root = Path(str(engine.get("root") or DEFAULT_ENGINE_ROOT)).resolve()
    files = build_job_files(job, settings)
    python = engine_python(engine)
    if engine_type == "sd_scripts":
        script = engine_root / sd_scripts_train_script(job)
        args = build_sd_scripts_args(job, files, script)
    else:
        script = engine_root / "anima_train_network.py"
        args = [
            "-m",
            "accelerate.commands.launch",
            *build_accelerate_flags(job),
            str(script),
            f"--config_file={files['mergedConfig']}",
        ]
    env = process_env_for_job(job)
    env["PYTHONPATH"] = str(engine_root)
    plan = make_plan(f"train:{job['name']}", "train", engine_root, python, args, env, files)
    errors = []
    if not engine_root.exists():
        errors.append(f"engine root not found: {engine_root}")
    if not script.exists():
        errors.append(f"training script not found: {script}")
    errors.extend(model_path_errors(job, engine_type))
    if engine_type == "sd_scripts":
        errors.extend(sd_scripts_option_errors(job))
    if python == sys.executable and engine_type == "sd_scripts":
        ok, _, stderr = run_command([python, "-c", "import accelerate"], timeout=10.0)
        if not ok:
            errors.append(f"python environment does not have accelerate installed: {python} {stderr}")
    dataset_dir = str(job.get("dataset", {}).get("imageDir", "")).strip()
    if not dataset_dir:
        errors.append("dataset imageDir is empty")
    elif not Path(dataset_dir).exists():
        errors.append(f"dataset imageDir not found: {dataset_dir}")
    if errors and not payload.get("allowInvalid"):
        return failure("train_launch_plan", errors, {"plan": plan})
    return success("train_launch_plan", {"job": job, "plan": plan, "errors": errors})


def find_free_port(start: int = 6006) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free local port found")


def tensorboard_launch_plan(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        return failure("tensorboard_launch_plan", ["name is required"])
    settings = load_settings_dict()
    job = load_job_dict(name, settings)
    engine = get_engine(settings, job.get("engineId"))
    files = build_job_files(job, settings)
    python = engine_python(engine)
    port = int(payload.get("port") or find_free_port(6006))
    args = ["-m", "tensorboard.main", f"--logdir={files['logsDir']}", f"--port={port}", "--host=127.0.0.1"]
    url = f"http://127.0.0.1:{port}"
    plan = make_plan(f"tensorboard:{job['name']}", "tensorboard", Path(files["root"]), python, args, {"PYTHONIOENCODING": "utf-8"}, files, url)
    return success("tensorboard_launch_plan", {"job": job, "plan": plan})


def latest_safetensors(output_dir: Path) -> str:
    files = sorted(output_dir.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(files[0]) if files else ""


def sample_launch_plan(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        return failure("sample_launch_plan", ["name is required"])
    settings = load_settings_dict()
    job = load_job_dict(name, settings)
    engine = get_engine(settings, job.get("engineId"))
    engine_type = str(engine.get("type") or "anima_standalone")
    engine_root = Path(str(engine.get("root") or DEFAULT_ENGINE_ROOT)).resolve()
    files = build_job_files(job, settings)
    python = engine_python(engine)
    model = job.get("modelPaths", {})
    sample = job.get("sample", {})
    network_weights = str(payload.get("networkWeights") or sample.get("networkWeights") or latest_safetensors(Path(files["outputDir"])))
    if engine_type == "sd_scripts":
        script = engine_root / sd_scripts_gen_script(job)
        args = build_sd_scripts_sample_args(job, files, script, network_weights)
    else:
        script = engine_root / "anima_gen.py"
        args = [
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process",
            "1",
            str(script),
            f"--dit_path={model.get('ditPath', '')}",
            f"--qwen3_path={model.get('qwen3Path', '')}",
            f"--vae_path={model.get('vaePath', '')}",
            f"--sample_prompts={files['samplePrompts']}",
            f"--output_dir={files['samplesDir']}",
            f"--output_name={job['name']}_sample",
            f"--mixed_precision={job.get('training', {}).get('mixedPrecision', 'bf16')}",
            f"--seed={job.get('training', {}).get('seed', 42)}",
        ]
        if network_weights:
            args += [f"--network_weights={network_weights}", f"--network_mul={sample.get('networkMul', 1.0)}"]
        if len(split_gpu_ids(sample.get("gpuIds", ""))) > 1:
            args.append(f"--device_map={sample.get('multiGpuMode', 'parallel_cfg')}")
    env = {"PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = str(engine_root)
    if sample.get("gpuIds"):
        env["CUDA_VISIBLE_DEVICES"] = sample["gpuIds"]
    plan = make_plan(f"sample:{job['name']}", "sample", engine_root, python, args, env, files)
    errors = []
    if not script.exists():
        errors.append(f"sample script not found: {script}")
    errors.extend(model_path_errors(job, engine_type))
    if not read_text(Path(files["samplePrompts"])).strip():
        errors.append("sample prompts are empty")
    if errors and not payload.get("allowInvalid"):
        return failure("sample_launch_plan", errors, {"plan": plan})
    return success("sample_launch_plan", {"job": job, "plan": plan, "errors": errors})


def tagger_model_status(_: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings_dict()
    model_dir = Path(str(settings.get("taggerModelDir") or DEFAULT_TAGGER_MODEL_DIR)).expanduser().resolve()
    files = {
        "model": str(model_dir / "model.onnx"),
        "tags": str(model_dir / "selected_tags.csv"),
    }
    return success(
        "tagger_model_status",
        {
            "modelDir": str(model_dir),
            "modelExists": Path(files["model"]).exists(),
            "tagsExists": Path(files["tags"]).exists(),
            "files": files,
        },
    )


def tagger_download_plan(payload: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings_dict()
    model_dir = Path(str(payload.get("modelDir") or settings.get("taggerModelDir") or DEFAULT_TAGGER_MODEL_DIR)).expanduser().resolve()
    args = ["-m", "lora_training_gui.wd14_tagger", "--model-dir", str(model_dir), "--download-only"]
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(PROJECT_ROOT / "python")}
    plan = make_plan("tagger:download", "tagger_download", PROJECT_ROOT, sys.executable, args, env, {"modelDir": str(model_dir)})
    return success("tagger_download_plan", {"plan": plan})


def tagger_install_deps_plan(_: dict[str, Any]) -> dict[str, Any]:
    args = ["-m", "pip", "install", "--upgrade", "pillow", "numpy", "onnxruntime"]
    plan = make_plan("tagger:install_deps", "tagger_install_deps", PROJECT_ROOT, sys.executable, args, {"PYTHONIOENCODING": "utf-8"})
    return success("tagger_install_deps_plan", {"plan": plan})


def tagger_launch_plan(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_root = str(payload.get("datasetRoot") or "").strip()
    if not dataset_root:
        return failure("tagger_launch_plan", ["datasetRoot is required"])
    settings = load_settings_dict()
    model_dir = Path(str(payload.get("modelDir") or settings.get("taggerModelDir") or DEFAULT_TAGGER_MODEL_DIR)).expanduser().resolve()
    args = [
        "-m",
        "lora_training_gui.wd14_tagger",
        "--model-dir",
        str(model_dir),
        "--dataset-root",
        dataset_root,
        "--caption-extension",
        str(payload.get("captionExtension") or ".txt"),
        "--threshold",
        str(payload.get("threshold") or 0.35),
        "--character-threshold",
        str(payload.get("characterThreshold") or 0.35),
        "--mode",
        str(payload.get("mode") or "merge"),
    ]
    if payload.get("recursive", True):
        args.append("--recursive")
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(PROJECT_ROOT / "python")}
    process_id = f"tagger:{sanitize_name(Path(dataset_root).name)}"
    plan = make_plan(process_id, "tagger", PROJECT_ROOT, sys.executable, args, env, {"modelDir": str(model_dir), "datasetRoot": dataset_root})
    return success("tagger_launch_plan", {"plan": plan})


def split_tags(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    if "," in cleaned:
        return [tag.strip() for tag in cleaned.split(",") if tag.strip()]
    return [tag.strip() for tag in cleaned.splitlines() if tag.strip()]


def join_tags(tags: list[str]) -> str:
    return ", ".join(tag.strip() for tag in tags if tag.strip())


def tag_matches(left: str, right: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return left.strip() == right.strip()
    return left.strip().lower() == right.strip().lower()


def get_png_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def get_gif_size(data: bytes) -> tuple[int, int] | None:
    if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    return None


def get_bmp_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"BM") and len(data) >= 26:
        width = struct.unpack("<I", data[18:22])[0]
        height = abs(struct.unpack("<i", data[22:26])[0])
        return width, height
    return None


def get_jpeg_size(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None

    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2

        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(data):
            return None

        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            return None

        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = struct.unpack(">H", data[index + 3 : index + 5])[0]
            width = struct.unpack(">H", data[index + 5 : index + 7])[0]
            return width, height

        index += segment_length

    return None


def get_image_size(path: Path) -> tuple[int, int] | None:
    try:
        data = path.read_bytes()[:256 * 1024]
    except OSError:
        return None

    for reader in (get_png_size, get_gif_size, get_bmp_size, get_jpeg_size):
        size = reader(data)
        if size:
            return size

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def iter_dataset_files(root: Path) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    captions: list[Path] = []

    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if name not in SKIP_DIRS]
        current = Path(current_root)
        for file_name in file_names:
            path = current / file_name
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                images.append(path)
            elif suffix == ".txt":
                captions.append(path)

    return sorted(images, key=lambda p: str(p).lower()), sorted(captions, key=lambda p: str(p).lower())


def build_dataset_item(root: Path, image_path: Path, caption_extension: str, min_pixels: int) -> dict[str, Any]:
    caption_path = image_path.with_suffix(caption_extension)
    caption_exists = caption_path.exists()
    caption_text = read_text(caption_path) if caption_exists else ""
    tags = split_tags(caption_text)
    size = get_image_size(image_path)
    issues: list[str] = []

    if not caption_exists:
        issues.append("missing_caption")
    elif not caption_text.strip():
        issues.append("empty_caption")

    width = height = None
    if size:
        width, height = size
        if min_pixels > 0 and width * height < min_pixels:
            issues.append("low_resolution")
    else:
        issues.append("image_size_unknown")

    return {
        "imagePath": str(image_path),
        "captionPath": str(caption_path),
        "relativePath": str(image_path.relative_to(root)),
        "fileName": image_path.name,
        "stem": image_path.stem,
        "captionExists": caption_exists,
        "captionText": caption_text,
        "tags": tags,
        "tagCount": len(tags),
        "width": width,
        "height": height,
        "issues": issues,
    }


def dataset_scan(payload: dict[str, Any]) -> dict[str, Any]:
    root_raw = str(payload.get("root") or "").strip()
    if not root_raw:
        return failure("dataset_scan", ["root is required"])

    root = Path(root_raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return failure("dataset_scan", [f"dataset root does not exist or is not a directory: {root}"])

    caption_extension = str(payload.get("captionExtension") or ".txt").strip()
    if not caption_extension.startswith("."):
        caption_extension = "." + caption_extension

    min_pixels = int(payload.get("minPixels") or 0)
    max_items = int(payload.get("maxItems") or 5000)

    images, captions = iter_dataset_files(root)
    image_caption_keys = {str(image.with_suffix(caption_extension)).lower() for image in images}
    orphan_captions = [caption for caption in captions if str(caption).lower() not in image_caption_keys]
    items = [build_dataset_item(root, image, caption_extension, min_pixels) for image in images[:max_items]]

    missing_caption_count = sum(1 for item in items if "missing_caption" in item["issues"])
    empty_caption_count = sum(1 for item in items if "empty_caption" in item["issues"])
    issue_count = sum(1 for item in items if item["issues"])

    return success(
        "dataset_scan",
        {
            "root": str(root),
            "captionExtension": caption_extension,
            "imageCount": len(images),
            "captionCount": len(captions),
            "shownItemCount": len(items),
            "truncated": len(images) > max_items,
            "missingCaptionCount": missing_caption_count,
            "emptyCaptionCount": empty_caption_count,
            "orphanCaptionCount": len(orphan_captions),
            "issueCount": issue_count,
            "items": items,
            "orphanCaptions": [
                {
                    "path": str(path),
                    "relativePath": str(path.relative_to(root)),
                    "text": read_text(path),
                }
                for path in orphan_captions[:1000]
            ],
        },
    )


def dataset_save_caption(payload: dict[str, Any]) -> dict[str, Any]:
    caption_raw = str(payload.get("captionPath") or "").strip()
    if not caption_raw:
        return failure("dataset_save_caption", ["captionPath is required"])

    caption_path = Path(caption_raw).expanduser().resolve()
    text = str(payload.get("text") or "")
    write_text(caption_path, text)
    tags = split_tags(text)

    return success(
        "dataset_save_caption",
        {
            "captionPath": str(caption_path),
            "captionText": text,
            "tags": tags,
            "tagCount": len(tags),
        },
    )


def apply_tag_operation_to_tags(tags: list[str], payload: dict[str, Any]) -> tuple[list[str], bool]:
    operation = str(payload.get("operation") or "").strip()
    tag = str(payload.get("tag") or "").strip()
    replacement = str(payload.get("replacement") or "").strip()
    case_sensitive = bool(payload.get("caseSensitive") or False)
    keep_tokens = max(0, int(payload.get("keepTokens") or 0))

    original = list(tags)

    if operation == "add":
        if tag and not any(tag_matches(existing, tag, case_sensitive) for existing in tags):
            tags.append(tag)
    elif operation == "remove":
        if tag:
            tags = [existing for existing in tags if not tag_matches(existing, tag, case_sensitive)]
    elif operation == "replace":
        if tag:
            tags = [replacement if tag_matches(existing, tag, case_sensitive) else existing for existing in tags]
            tags = [existing for existing in tags if existing.strip()]
    elif operation == "move_front":
        if tag:
            matched = [existing for existing in tags if tag_matches(existing, tag, case_sensitive)]
            rest = [existing for existing in tags if not tag_matches(existing, tag, case_sensitive)]
            tags = matched + rest
    elif operation == "shuffle":
        locked = tags[:keep_tokens]
        movable = tags[keep_tokens:]
        random.shuffle(movable)
        tags = locked + movable
    else:
        raise ValueError(f"unsupported operation: {operation}")

    return tags, tags != original


def dataset_apply_tag_operation(payload: dict[str, Any]) -> dict[str, Any]:
    root_raw = str(payload.get("root") or "").strip()
    if not root_raw:
        return failure("dataset_apply_tag_operation", ["root is required"])

    root = Path(root_raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return failure("dataset_apply_tag_operation", [f"dataset root does not exist: {root}"])

    caption_extension = str(payload.get("captionExtension") or ".txt").strip()
    if not caption_extension.startswith("."):
        caption_extension = "." + caption_extension

    include_missing = bool(payload.get("includeMissing") or False)
    selected_paths_raw = payload.get("selectedImagePaths") or []
    selected_paths = {str(Path(path).expanduser().resolve()).lower() for path in selected_paths_raw if str(path).strip()}

    images, _ = iter_dataset_files(root)
    changed: list[dict[str, Any]] = []
    skipped = 0

    for image in images:
        if selected_paths and str(image.resolve()).lower() not in selected_paths:
            continue

        caption_path = image.with_suffix(caption_extension)
        if not caption_path.exists() and not include_missing:
            skipped += 1
            continue

        text = read_text(caption_path) if caption_path.exists() else ""
        tags = split_tags(text)
        next_tags, did_change = apply_tag_operation_to_tags(tags, payload)
        next_text = join_tags(next_tags)
        if did_change or (include_missing and not caption_path.exists()):
            write_text(caption_path, next_text)
            changed.append(
                {
                    "imagePath": str(image),
                    "captionPath": str(caption_path),
                    "captionText": next_text,
                    "tags": next_tags,
                    "tagCount": len(next_tags),
                }
            )
        else:
            skipped += 1

    return success(
        "dataset_apply_tag_operation",
        {
            "root": str(root),
            "changedCount": len(changed),
            "skippedCount": skipped,
            "changed": changed,
        },
    )


def dataset_find_tags(payload: dict[str, Any]) -> dict[str, Any]:
    root_raw = str(payload.get("root") or "").strip()
    query = str(payload.get("query") or "").strip()
    if not root_raw:
        return failure("dataset_find_tags", ["root is required"])
    if not query:
        return failure("dataset_find_tags", ["query is required"])

    root = Path(root_raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return failure("dataset_find_tags", [f"dataset root does not exist: {root}"])

    case_sensitive = bool(payload.get("caseSensitive") or False)
    use_regex = bool(payload.get("regex") or False)
    pattern = re.compile(query if use_regex else re.escape(query), 0 if case_sensitive else re.IGNORECASE)

    matches: list[dict[str, Any]] = []
    _, captions = iter_dataset_files(root)
    for caption in captions:
        text = read_text(caption)
        tags = split_tags(text)
        hit_tags = [tag for tag in tags if pattern.search(tag)]
        if hit_tags:
            matches.append(
                {
                    "captionPath": str(caption),
                    "relativePath": str(caption.relative_to(root)),
                    "tags": hit_tags,
                    "tagCount": len(hit_tags),
                }
            )

    return success("dataset_find_tags", {"root": str(root), "matches": matches})


def health_check(_: dict[str, Any]) -> dict[str, Any]:
    data = {
        "app": {
            "projectRoot": str(PROJECT_ROOT),
            "bridgeRoot": str(Path(__file__).resolve().parent),
        },
        "system": {
            "platform": platform.platform(),
            "pythonVersion": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "cwd": os.getcwd(),
        },
        "tools": [
            detect_tool("Git", ["git", "--version"]),
            detect_tool("Node.js", ["node", "--version"]),
            detect_tool("pnpm", ["pnpm", "--version"]),
            detect_tool("Cargo", ["cargo", "--version"]),
            detect_tool("nvidia-smi", ["nvidia-smi", "--version"]),
        ],
        "gpu": detect_gpus(),
        "engines": detect_engines(),
    }
    return success("health_check", data)


def detect_gpus_job(_: dict[str, Any]) -> dict[str, Any]:
    return success("detect_gpus", detect_gpus())


JOBS = {
    "settings_get": settings_get,
    "settings_save": settings_save,
    "health_check": health_check,
    "detect_gpus": detect_gpus_job,
    "dataset_scan": dataset_scan,
    "dataset_save_caption": dataset_save_caption,
    "dataset_apply_tag_operation": dataset_apply_tag_operation,
    "dataset_find_tags": dataset_find_tags,
    "jobs_list": jobs_list,
    "job_create": job_create,
    "job_get": job_get,
    "job_save": job_save,
    "job_clone": job_clone,
    "job_delete": job_delete,
    "job_build_files": job_build_files,
    "train_launch_plan": train_launch_plan,
    "tensorboard_launch_plan": tensorboard_launch_plan,
    "sample_launch_plan": sample_launch_plan,
    "tagger_model_status": tagger_model_status,
    "tagger_download_plan": tagger_download_plan,
    "tagger_install_deps_plan": tagger_install_deps_plan,
    "tagger_launch_plan": tagger_launch_plan,
}


def parse_payload(raw_json: str | None, payload_path: str | None = None) -> dict[str, Any]:
    if payload_path:
        raw_json = read_text(Path(payload_path).expanduser().resolve())

    if not raw_json:
        return {}

    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("payload must be a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--payload")
    parser.add_argument("--payload-json", default="{}")
    args = parser.parse_args()

    try:
        payload = parse_payload(args.payload_json, args.payload)
        handler = JOBS.get(args.job)
        if not handler:
            result = failure(args.job, [f"unknown bridge job: {args.job}"])
        else:
            result = handler(payload)
    except Exception as error:  # noqa: BLE001 - bridge must serialize failures.
        result = failure(args.job, [f"{type(error).__name__}: {error}"])

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
