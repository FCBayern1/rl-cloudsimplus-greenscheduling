#!/usr/bin/env python3
"""
Normalize a set of plot images to the same final size for paper layout.

Typical workflow:
  1) Trim surrounding whitespace (optional)
  2) Normalize canvas sizes either by padding (recommended) or center-cropping

Why padding is recommended:
  - avoids cutting axis labels/titles
  - guarantees identical output dimensions

Usage:
  /path/to/.venv/bin/python drl-manager/scripts/normalize_plot_images.py \\
    --mode pad --trim \\
    --scale-to width \\
    --out-dir drl-manager/compare_result/normalized_figs \\
    /abs/path/a.png /abs/path/b.png /abs/path/c.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageChops


def trim_whitespace(im: Image.Image, bg: Tuple[int, int, int, int] | None = None) -> Image.Image:
    """
    Trim whitespace by comparing with a solid background.
    Defaults to using the top-left pixel as background (works for white).
    """
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    if bg is None:
        bg = im.getpixel((0, 0))
    bg_im = Image.new("RGBA", im.size, bg)
    diff = ImageChops.difference(im, bg_im)
    bbox = diff.getbbox()
    if bbox is None:
        return im
    return im.crop(bbox)

def resize_keep_aspect(im: Image.Image, *, target_w: int | None, target_h: int | None) -> Image.Image:
    """
    Resize while keeping aspect ratio. Provide exactly one of target_w/target_h.
    """
    if (target_w is None) == (target_h is None):
        raise ValueError("Provide exactly one of target_w or target_h.")
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    w, h = im.size
    if target_w is not None:
        new_w = int(target_w)
        new_h = max(1, int(round(h * (new_w / w))))
    else:
        new_h = int(target_h)
        new_w = max(1, int(round(w * (new_h / h))))
    return im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def pad_to(im: Image.Image, size: Tuple[int, int], bg=(255, 255, 255, 255)) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    w, h = im.size
    W, H = size
    canvas = Image.new("RGBA", (W, H), bg)
    x = (W - w) // 2
    y = (H - h) // 2
    canvas.paste(im, (x, y))
    return canvas


def center_crop_to(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    w, h = im.size
    W, H = size
    if W > w or H > h:
        raise ValueError(f"Cannot center-crop to larger size {size} from {im.size}")
    left = (w - W) // 2
    top = (h - H) // 2
    return im.crop((left, top, left + W, top + H))


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize plot PNG sizes (trim + pad/crop).")
    p.add_argument("images", nargs="+", help="Input image paths (png/jpg). Use absolute paths.")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--mode", choices=["pad", "crop"], default="pad", help="pad (recommended) or crop")
    p.add_argument("--trim", action="store_true", help="Trim surrounding whitespace before normalizing")
    p.add_argument(
        "--scale-to",
        choices=["none", "width", "height"],
        default="none",
        help="Scale content to a unified width/height before pad/crop (recommended: width)",
    )
    p.add_argument("--target-width", type=int, default=0, help="If set (>0), scale-to width uses this value")
    p.add_argument("--target-height", type=int, default=0, help="If set (>0), scale-to height uses this value")
    p.add_argument("--suffix", default="_norm", help="Output filename suffix before extension")
    args = p.parse_args()

    in_paths = [Path(x) for x in args.images]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ims: List[Image.Image] = []
    sizes: List[Tuple[int, int]] = []
    for ip in in_paths:
        im = Image.open(ip)
        if args.trim:
            im = trim_whitespace(im)
        ims.append(im)
        sizes.append(im.size)

    # Step 1: optional scaling of content to a unified width/height
    if args.scale_to != "none":
        if args.scale_to == "width":
            target_w = args.target_width if args.target_width and args.target_width > 0 else max(w for w, _ in sizes)
            ims = [resize_keep_aspect(im, target_w=target_w, target_h=None) for im in ims]
        else:
            target_h = args.target_height if args.target_height and args.target_height > 0 else max(h for _, h in sizes)
            ims = [resize_keep_aspect(im, target_w=None, target_h=target_h) for im in ims]
        sizes = [im.size for im in ims]

    if args.mode == "pad":
        target = (max(w for w, _ in sizes), max(h for _, h in sizes))
    else:
        target = (min(w for w, _ in sizes), min(h for _, h in sizes))

    for ip, im in zip(in_paths, ims):
        if args.mode == "pad":
            out_im = pad_to(im, target)
        else:
            out_im = center_crop_to(im, target)
        # Convert to RGB to avoid alpha issues in some LaTeX pipelines
        out_im = out_im.convert("RGB")
        out_path = out_dir / f"{ip.stem}{args.suffix}{ip.suffix}"
        out_im.save(out_path)
        print(f"Saved: {out_path}  size={target}")


if __name__ == "__main__":
    main()


