from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_BASE_URL = "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main"
MODEL_FILES = {
    "model.onnx": f"{MODEL_BASE_URL}/model.onnx",
    "selected_tags.csv": f"{MODEL_BASE_URL}/selected_tags.csv",
}


def log(message: str) -> None:
    print(message, flush=True)


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        log(f"exists: {target}")
        return
    log(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            done += len(chunk)
            if total:
                log(f"downloaded {done / total:.1%}: {target.name}")
            else:
                log(f"downloaded {done} bytes: {target.name}")


def ensure_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for file_name, url in MODEL_FILES.items():
        download_file(url, model_dir / file_name)


def read_caption(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [part.strip() for part in text.splitlines() if part.strip()]


def write_caption(path: Path, tags: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(", ".join(tags), encoding="utf-8")


def iter_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_tags(csv_path: Path) -> list[tuple[str, int]]:
    tags: list[tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").replace("_", " ").strip()
            if not name:
                continue
            try:
                category = int(row.get("category") or 0)
            except ValueError:
                category = 0
            tags.append((name, category))
    return tags


def prepare_image(image_path: Path, size: int):
    try:
        from PIL import Image
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Missing dependencies. Run Tagger dependency install first.") from error

    image = Image.open(image_path).convert("RGB")
    canvas_size = max(image.size)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    offset = ((canvas_size - image.width) // 2, (canvas_size - image.height) // 2)
    canvas.paste(image, offset)
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32)
    array = array[:, :, ::-1]
    return array[None, :, :, :]


def tag_image(session, tags: list[tuple[str, int]], image_path: Path, threshold: float, character_threshold: float) -> list[str]:
    inputs = session.get_inputs()
    input_name = inputs[0].name
    shape = inputs[0].shape
    size = 448
    for dim in shape:
        if isinstance(dim, int) and dim > 32:
            size = dim
            break
    tensor = prepare_image(image_path, size)
    outputs = session.run(None, {input_name: tensor})
    scores = outputs[0][0]
    selected: list[str] = []
    for (tag, category), score in zip(tags, scores):
        score_float = float(score)
        if category == 9:
            continue
        cutoff = character_threshold if category == 4 else threshold
        if score_float >= cutoff:
            selected.append(tag)
    return selected


def run_tagger(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).expanduser().resolve()
    ensure_model(model_dir)
    if args.download_only:
        return 0

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Missing onnxruntime. Run Tagger dependency install first.") from error

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    tags = load_tags(model_dir / "selected_tags.csv")
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    images = iter_images(dataset_root, args.recursive)
    log(f"tagging {len(images)} images")

    for index, image_path in enumerate(images, start=1):
        detected = tag_image(session, tags, image_path, args.threshold, args.character_threshold)
        caption_path = image_path.with_suffix(args.caption_extension)
        if args.mode == "overwrite":
            next_tags = detected
        else:
            existing = read_caption(caption_path)
            seen = {tag.lower(): tag for tag in existing}
            next_tags = list(existing)
            for tag in detected:
                if tag.lower() not in seen:
                    next_tags.append(tag)
                    seen[tag.lower()] = tag
        write_caption(caption_path, next_tags)
        log(f"[{index}/{len(images)}] {image_path.name}: {len(detected)} tags")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--caption-extension", default=".txt")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--character-threshold", type=float, default=0.35)
    parser.add_argument("--mode", choices=["merge", "overwrite"], default="merge")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not args.caption_extension.startswith("."):
        args.caption_extension = "." + args.caption_extension
    return run_tagger(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
