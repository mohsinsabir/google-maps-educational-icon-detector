#!/usr/bin/env python3
"""Detect educational institute icons in a Google Maps screenshot.

Uses multi-scale OpenCV template matching (TM_CCOEFF_NORMED) plus
non-maximum suppression. No machine learning is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Literal, Sequence, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

MatchMode = Literal["grayscale", "edges", "auto"]
PrepareMode = Literal["grayscale", "edges", "color"]
ImageArray = NDArray[np.uint8]


class Detection(TypedDict):
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float


# Slight blur before Canny reduces map-texture noise without wiping the icon.
_EDGE_BLUR_KERNEL = (3, 3)
_CANNY_LOW = 50
_CANNY_HIGH = 150
_MIN_TEMPLATE_SIDE = 8
# Whole-pin matching hits every Google Maps marker. The inner disc + glyph
# correlation is what separates a graduation cap from a shopping bag.
_INNER_THRESHOLD = 0.88
_INNER_SAT_MIN = 35
_TEXT_SCALE = 0.5
_TEXT_THICKNESS = 1
_BOX_COLOR = (0, 255, 0)
_LABEL_TEXT_COLOR = (255, 255, 255)
_BOX_THICKNESS = 2
_MARKER_RADIUS = 3
_LABEL = "school"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path: str | Path) -> ImageArray:
    """Load an image from disk as a BGR (or grayscale) uint8 array.

    Accepts PNG/JPEG and both color and grayscale files. 16-bit images
    are converted to 8-bit. Raises FileNotFoundError or ValueError.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image (invalid or unsupported): {path}")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError(f"Image has no usable pixels: {path}")

    return image


def _to_gray(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_bgr(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _to_edges(gray: ImageArray) -> ImageArray:
    blurred = cv2.GaussianBlur(gray, _EDGE_BLUR_KERNEL, 0)
    return cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)


def _prepare_pair(
    image: ImageArray,
    template: ImageArray,
    mode: PrepareMode,
) -> tuple[ImageArray, ImageArray]:
    if mode == "color":
        return _to_bgr(image), _to_bgr(template)
    gray_image = _to_gray(image)
    gray_template = _to_gray(template)
    if mode == "edges":
        return _to_edges(gray_image), _to_edges(gray_template)
    return gray_image, gray_template


def _icon_interior_mask(template: ImageArray) -> ImageArray:
    """Mask the colored pin disc, including the white glyph inside it."""
    bgr = _to_bgr(template)
    saturation = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    fill = np.where(saturation >= _INNER_SAT_MIN, 255, 0).astype(np.uint8)
    if cv2.countNonZero(fill) < 20:
        gray = _to_gray(bgr)
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        background = float(np.median(border))
        fill = (np.abs(gray.astype(np.int16) - background) > 25).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(cv2.morphologyEx(fill, cv2.MORPH_CLOSE, kernel), kernel)
    if cv2.countNonZero(mask) == 0:
        return np.full(template.shape[:2], 255, dtype=np.uint8)
    return mask


def _ccoeff_normed(
    patch: ImageArray,
    template: ImageArray,
    mask: ImageArray | None = None,
) -> float:
    """Same score as cv2.TM_CCOEFF_NORMED, optionally on masked pixels."""
    if patch.shape != template.shape or patch.size == 0:
        return 0.0
    if mask is None:
        pixels = patch.astype(np.float64).ravel()
        templ = template.astype(np.float64).ravel()
    else:
        keep = mask > 0
        if patch.ndim == 3:
            keep = np.repeat(keep[:, :, None], patch.shape[2], axis=2)
        if np.count_nonzero(keep) < 8:
            return 0.0
        pixels = patch[keep].astype(np.float64)
        templ = template[keep].astype(np.float64)

    pixels -= pixels.mean()
    templ -= templ.mean()
    denom = float(np.linalg.norm(pixels) * np.linalg.norm(templ))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(pixels, templ) / denom)


def filter_by_icon_interior(
    image: ImageArray,
    template: ImageArray,
    detections: Sequence[Detection],
    inner_threshold: float = _INNER_THRESHOLD,
) -> list[Detection]:
    """Drop pins whose interior symbol does not match the template glyph."""
    gray_image = _to_gray(image)
    gray_template = _to_gray(template)
    mask = _icon_interior_mask(template)
    img_h, img_w = gray_image.shape[:2]
    kept: list[Detection] = []

    for det in detections:
        x, y, width, height = det["x"], det["y"], det["width"], det["height"]
        if x < 0 or y < 0 or x + width > img_w or y + height > img_h:
            continue
        patch = gray_image[y : y + height, x : x + width]
        scaled_template = cv2.resize(gray_template, (width, height), interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if _ccoeff_normed(patch, scaled_template, scaled_mask) >= inner_threshold:
            kept.append(det)
    return kept


def _scale_range(min_scale: float, max_scale: float, scale_step: float) -> list[float]:
    if scale_step <= 0:
        raise ValueError("scale_step must be > 0")
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("scales must be > 0")
    if min_scale > max_scale:
        raise ValueError("min_scale must be <= max_scale")

    scales: list[float] = []
    scale = min_scale
    # Inclusive upper bound with a small epsilon for float accumulation.
    while scale <= max_scale + 1e-9:
        scales.append(round(scale, 5))
        scale += scale_step
    return scales


def _resize_template(template: ImageArray, scale: float) -> ImageArray | None:
    height, width = template.shape[:2]
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if new_w < _MIN_TEMPLATE_SIDE or new_h < _MIN_TEMPLATE_SIDE:
        return None
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(template, (new_w, new_h), interpolation=interpolation)


def _local_maxima(score_map: NDArray[np.floating], threshold: float) -> tuple[NDArray, NDArray]:
    """Keep only 3x3 peaks at or above the confidence threshold."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(score_map, kernel)
    peak_mask = (score_map >= threshold) & (score_map >= dilated)
    return np.where(peak_mask)


def match_template_multiscale(
    image: ImageArray,
    template: ImageArray,
    threshold: float = 0.80,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    scale_step: float = 0.05,
) -> list[Detection]:
    """Run TM_CCOEFF_NORMED matching across a range of template scales.

    The screenshot is left at native resolution; the template is resized.
    Scales where the template would be larger than the screenshot are skipped.
    """
    img_h, img_w = image.shape[:2]
    detections: list[Detection] = []

    for scale in _scale_range(min_scale, max_scale, scale_step):
        scaled = _resize_template(template, scale)
        if scaled is None:
            continue

        t_h, t_w = scaled.shape[:2]
        if t_h > img_h or t_w > img_w:
            continue

        score_map = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
        ys, xs = _local_maxima(score_map, threshold)
        for y, x in zip(ys, xs):
            detections.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(t_w),
                    "height": int(t_h),
                    "center_x": int(x + t_w // 2),
                    "center_y": int(y + t_h // 2),
                    "confidence": float(score_map[y, x]),
                }
            )

    return detections


def _iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    inter_w = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    inter_h = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = inter_w * inter_h
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


def non_max_suppression(
    detections: Sequence[Detection],
    iou_threshold: float = 0.35,
) -> list[Detection]:
    """Drop overlapping boxes, keeping the highest-confidence detection."""
    remaining = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[Detection] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if _iou(best, d) < iou_threshold]
    return kept


def draw_detections(image: ImageArray, detections: Iterable[Detection]) -> ImageArray:
    """Draw green boxes with a 'school' tag and center markers."""
    canvas = _to_bgr(image)
    pad_x, pad_y = 4, 3
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        cx, cy = det["center_x"], det["center_y"]

        cv2.rectangle(canvas, (x, y), (x + w, y + h), _BOX_COLOR, _BOX_THICKNESS)
        cv2.circle(canvas, (cx, cy), _MARKER_RADIUS, _BOX_COLOR, thickness=-1)

        (text_w, text_h), baseline = cv2.getTextSize(
            _LABEL, cv2.FONT_HERSHEY_SIMPLEX, _TEXT_SCALE, _TEXT_THICKNESS
        )
        box_h = text_h + baseline + pad_y * 2
        box_w = text_w + pad_x * 2
        tag_y1 = y - box_h if y - box_h >= 0 else y
        tag_y2 = tag_y1 + box_h
        cv2.rectangle(canvas, (x, tag_y1), (x + box_w, tag_y2), _BOX_COLOR, thickness=-1)
        cv2.putText(
            canvas,
            _LABEL,
            (x + pad_x, tag_y2 - pad_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            _TEXT_SCALE,
            _LABEL_TEXT_COLOR,
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )
    return canvas


def detect_educational_icons(
    image: ImageArray,
    template: ImageArray,
    threshold: float = 0.80,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    scale_step: float = 0.05,
    mode: MatchMode = "auto",
    nms_iou: float = 0.35,
    inner_threshold: float = _INNER_THRESHOLD,
) -> list[Detection]:
    """Detect all educational institute icons and return NMS-filtered boxes."""
    img_h, img_w = image.shape[:2]
    tpl_h, tpl_w = template.shape[:2]
    smallest_h = max(1, int(round(tpl_h * min_scale)))
    smallest_w = max(1, int(round(tpl_w * min_scale)))
    if smallest_h > img_h or smallest_w > img_w:
        raise ValueError(
            "Template is larger than the screenshot even at min_scale "
            f"({smallest_w}x{smallest_h} vs {img_w}x{img_h}). "
            "Crop a smaller icon or lower --min-scale."
        )

    modes: tuple[PrepareMode, ...]
    if mode == "auto":
        modes = ("grayscale", "edges", "color")
    else:
        modes = (mode,)

    combined: list[Detection] = []
    for match_mode in modes:
        prepared_image, prepared_template = _prepare_pair(image, template, match_mode)
        combined.extend(
            match_template_multiscale(
                prepared_image,
                prepared_template,
                threshold=threshold,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_step=scale_step,
            )
        )

    combined = filter_by_icon_interior(
        image, template, combined, inner_threshold=inner_threshold
    )
    return non_max_suppression(combined, iou_threshold=nms_iou)


def _print_detections(detections: Sequence[Detection]) -> None:
    count = len(detections)
    if count == 0:
        print("No matching educational institute icons found.")
        print("[]")
        return

    noun = "icon" if count == 1 else "icons"
    print(f"Found {count} educational institute {noun}:\n")
    for i, det in enumerate(detections, start=1):
        print(
            f"  #{i}  x={det['x']}  y={det['y']}  "
            f"width={det['width']}  height={det['height']}  "
            f"center=({det['center_x']}, {det['center_y']})  "
            f"confidence={det['confidence']:.4f}"
        )
    print("\nJSON:")
    print(json.dumps(list(detections), indent=2))


def choose_images() -> list[Path]:
    """Open a file dialog so the user can add one or more screenshots."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilenames(
        title="Select Google Maps screenshot(s)",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return [Path(path) for path in selected]


def collect_image_paths(image_args: Sequence[str] | None) -> list[Path]:
    """Resolve CLI paths, a folder of images (recursive), or a file-picker selection."""
    if not image_args:
        paths = choose_images()
        if not paths:
            raise FileNotFoundError("No image selected.")
        return paths

    paths: list[Path] = []
    for raw in image_args:
        path = Path(raw)
        if path.is_dir():
            found = sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in _IMAGE_EXTS
            )
            if not found:
                raise FileNotFoundError(f"No images found in folder: {path}")
            paths.extend(found)
        else:
            paths.append(path)
    return paths


def _unique_output_name(image_path: Path, name_roots: Sequence[Path]) -> str:
    """Build a collision-safe filename (e.g. 741407_424082.png for tile grids)."""
    resolved = image_path.resolve()
    for root in name_roots:
        try:
            rel = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        if len(rel.parts) > 1:
            return "_".join(rel.parts)
        return rel.name
    parent = image_path.parent.name
    if parent:
        return f"{parent}_{image_path.name}"
    return image_path.name


def output_path_for(
    image_path: Path,
    output_root: str | Path,
    has_detections: bool,
    name_roots: Sequence[Path] | None = None,
) -> Path:
    """Route annotated images into output/detected or output/not_detected."""
    root = Path(output_root)
    bucket = "detected" if has_detections else "not_detected"
    filename = _unique_output_name(image_path, name_roots or ())
    return root / bucket / filename


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect educational institute icons in a Google Maps screenshot."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Screenshot file(s) or a folder. Omit to choose files in a dialog.",
    )
    parser.add_argument(
        "--image",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Screenshot file(s) or a folder (same as positional images)",
    )
    parser.add_argument("--template", default="template1.png", help="Path to the cropped icon")
    parser.add_argument("--threshold", type=float, default=0.80, help="Match confidence threshold")
    parser.add_argument("--min-scale", type=float, default=0.7, help="Smallest template scale")
    parser.add_argument("--max-scale", type=float, default=1.3, help="Largest template scale")
    parser.add_argument("--scale-step", type=float, default=0.05, help="Template scale increment")
    parser.add_argument(
        "--mode",
        choices=("grayscale", "edges", "auto"),
        default="auto",
        help="Matching mode (default: auto)",
    )
    parser.add_argument(
        "--inner-threshold",
        type=float,
        default=_INNER_THRESHOLD,
        help="Minimum interior-glyph match; rejects other map pin types",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.35,
        help="IoU threshold for non-maximum suppression",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output root folder (writes to detected/ and not_detected/ inside it)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested = list(args.images) + list(args.image)

    try:
        image_paths = collect_image_paths(requested)
        template = load_image(args.template)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_root = Path(args.output)
    detected_dir = output_root / "detected"
    not_detected_dir = output_root / "not_detected"
    detected_dir.mkdir(parents=True, exist_ok=True)
    not_detected_dir.mkdir(parents=True, exist_ok=True)

    name_roots = [Path(raw) for raw in requested if Path(raw).is_dir()]

    exit_code = 0
    for image_path in image_paths:
        print(f"Processing: {image_path}")
        try:
            image = load_image(image_path)
            detections = detect_educational_icons(
                image,
                template,
                threshold=args.threshold,
                min_scale=args.min_scale,
                max_scale=args.max_scale,
                scale_step=args.scale_step,
                mode=args.mode,
                nms_iou=args.nms_iou,
                inner_threshold=args.inner_threshold,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        _print_detections(detections)

        output_path = output_path_for(
            image_path,
            output_root,
            has_detections=bool(detections),
            name_roots=name_roots,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, detections)
        if not cv2.imwrite(str(output_path), annotated):
            print(f"Error: could not write output image: {output_path}", file=sys.stderr)
            exit_code = 1
            continue

        print(f"\nAnnotated image saved to: {output_path}\n")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
