from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "trash"}


def log(message: str) -> None:
    print(message, flush=True)


def sanitize_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or "image"


def iter_images(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        results: list[Path] = []
        for current_root, dir_names, file_names in os.walk(root):
            dir_names[:] = [name for name in dir_names if name not in SKIP_DIRS]
            current = Path(current_root)
            for file_name in file_names:
                path = current / file_name
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    results.append(path)
        return sorted(results, key=lambda path: str(path).lower())
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path).lower(),
    )


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def target_path_for(source_root: Path, image_path: Path, dataset_root: Path, flatten: bool, output_format: str) -> Path:
    output_ext = image_path.suffix.lower()
    if output_format in {"png", "jpg", "jpeg"}:
        output_ext = ".jpg" if output_format in {"jpg", "jpeg"} else ".png"
    relative = image_path.relative_to(source_root)
    if flatten:
        stem = sanitize_part("_".join(relative.with_suffix("").parts))
        return dataset_root / f"{stem}{output_ext}"
    return dataset_root / relative.with_suffix(output_ext)


def copy_caption(image_path: Path, target_image: Path, caption_extension: str, write_empty: bool) -> bool:
    source_caption = image_path.with_suffix(caption_extension)
    target_caption = target_image.with_suffix(caption_extension)
    target_caption.parent.mkdir(parents=True, exist_ok=True)
    if source_caption.exists():
        shutil.copy2(source_caption, target_caption)
        return True
    if write_empty and not target_caption.exists():
        target_caption.write_text("", encoding="utf-8")
    return False


def process_with_pillow(image_path: Path, target_path: Path, max_side: int, output_format: str) -> dict[str, Any]:
    from PIL import Image, ImageOps

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened)
        original_size = image.size
        if max_side > 0:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if output_format in {"jpg", "jpeg"}:
            image = image.convert("RGB")
            save_format = "JPEG"
            save_kwargs: dict[str, Any] = {"quality": 95, "subsampling": 0}
        elif output_format == "png" or target_path.suffix.lower() == ".png":
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            save_format = "PNG"
            save_kwargs = {"optimize": True}
        else:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            save_format = image.format or None
            save_kwargs = {}
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_path, format=save_format, **save_kwargs)
        return {
            "originalWidth": original_size[0],
            "originalHeight": original_size[1],
            "width": image.size[0],
            "height": image.size[1],
            "converted": True,
        }


def process_image(image_path: Path, target_path: Path, max_side: int, output_format: str) -> dict[str, Any]:
    should_convert = output_format != "keep" or max_side > 0 or image_path.suffix.lower() in {".webp", ".bmp", ".gif"}
    if should_convert:
        try:
            return process_with_pillow(image_path, target_path, max_side, output_format)
        except ImportError:
            log("Pillow is not installed; falling back to file copy.")
        except Exception as error:
            log(f"Image conversion failed for {image_path.name}: {type(error).__name__}: {error}; copying original.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.resolve() != target_path.resolve():
        shutil.copy2(image_path, target_path)
    return {"converted": False}


def run(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not args.caption_extension.startswith("."):
        args.caption_extension = "." + args.caption_extension
    if args.output_format not in {"keep", "png", "jpg", "jpeg"}:
        raise ValueError(f"unsupported output format: {args.output_format}")

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    images = iter_images(source_root, args.recursive)
    seen_hashes: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    duplicate_count = 0
    caption_copied_count = 0
    log(f"preprocessing {len(images)} images")

    for index, image_path in enumerate(images, start=1):
        digest = file_sha1(image_path)
        duplicate_of = seen_hashes.get(digest)
        if duplicate_of:
            duplicate_count += 1
        else:
            seen_hashes[digest] = str(image_path)
        target_image = target_path_for(source_root, image_path, dataset_root, args.flatten, args.output_format)
        details = process_image(image_path, target_image, args.max_side, args.output_format)
        caption_copied = copy_caption(image_path, target_image, args.caption_extension, args.write_empty_captions)
        caption_copied_count += 1 if caption_copied else 0
        record = {
            "source": str(image_path),
            "target": str(target_image),
            "relativeSource": str(image_path.relative_to(source_root)),
            "sha1": digest,
            "duplicateOf": duplicate_of,
            "captionCopied": caption_copied,
            **details,
        }
        records.append(record)
        log(f"[{index}/{len(images)}] {image_path.name} -> {target_image.name}")

    manifest = {
        "schemaVersion": 1,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sourceRoot": str(source_root),
        "datasetRoot": str(dataset_root),
        "imageCount": len(images),
        "processedCount": len(records),
        "duplicateHashCount": duplicate_count,
        "captionCopiedCount": caption_copied_count,
        "captionExtension": args.caption_extension,
        "outputFormat": args.output_format,
        "maxSide": args.max_side,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"manifest: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--caption-extension", default=".txt")
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--output-format", choices=["keep", "png", "jpg", "jpeg"], default="png")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--write-empty-captions", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
