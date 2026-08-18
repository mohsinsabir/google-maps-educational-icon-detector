# Google Maps Educational Icon Detector

Detect **school / educational institute** map pins in Google Maps screenshots using classical computer vision — no machine learning, no GPU, no training data.

The tool crops a single reference icon (template), then finds every matching pin in one or more screenshots with multi-scale OpenCV template matching, interior-glyph filtering (so shopping bags and hospitals don’t get labeled as schools), and non-maximum suppression.

---

## Highlights

| Capability | Detail |
|---|---|
| **Zero ML** | Pure OpenCV + NumPy (`TM_CCOEFF_NORMED`) |
| **Scale robust** | Template is resized across a configurable scale range |
| **False-positive resistant** | Interior disc + glyph correlation rejects other map pin types |
| **Multi-mode matching** | Grayscale, edge (Canny), and color — or `auto` for all three |
| **Batch friendly** | Single file, folder, or GUI file picker |
| **Structured output** | Annotated PNG + JSON-ready detection list to stdout |

---

## Demo

```text
$ python detect_icon.py ../test-images/test.png --template template1.png

Processing: ../test-images/test.png
Found 3 educational institute icons:

  #1  x=398  y=110  width=32  height=37  center=(414, 128)  confidence=0.9932
  #2  x=597  y=382  width=32  height=37  center=(613, 400)  confidence=0.9344
  #3  x=782  y=468  width=32  height=37  center=(798, 486)  confidence=0.9219

Annotated image saved to: output/detected.png
```

Annotated results land in `google-map-icon-detector/output/` with green bounding boxes, center markers, and a `school` label.

---

## Project structure

```text
School_Detection/
├── README.md
├── test-images/                    # Sample Google Maps screenshots
│   ├── test.png
│   ├── test2.png
│   ├── test3.png
│   ├── test4.png
│   ├── input.png
│   └── hospitaltest.png            # Negative / mixed-pin control
└── google-map-icon-detector/
    ├── detect_icon.py              # Main CLI + detection pipeline
    ├── requirements.txt
    ├── template1.png               # Default cropped school pin
    ├── template2.png               # Alternate template
    └── output/
        ├── detected.png            # Annotated result(s)
        └── .gitkeep
```

---

## Requirements

- **Python** 3.10+ (typed with `from __future__ import annotations`)
- **OpenCV** (`opencv-python`)
- **NumPy**
- **Tkinter** (optional — only for the file-picker when no image path is given; usually bundled with Python on Windows)

---

## Setup

```bash
cd google-map-icon-detector

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`:

```text
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## Quick start

From `google-map-icon-detector/`:

```bash
# GUI: pick screenshot(s) interactively (uses template1.png by default)
python detect_icon.py

# Single screenshot
python detect_icon.py ../test-images/test.png

# Custom template + output path
python detect_icon.py ../test-images/test2.png \
  --template template2.png \
  --output output/schools.png

# Entire folder of screenshots
python detect_icon.py ../test-images --output output/detected.png
```

When multiple images are processed, outputs are named `{stem}_detected.png` next to the `--output` path (for example `test_detected.png`, `test2_detected.png`).

---

## Usage

### Image inputs

| Input style | Behavior |
|---|---|
| No arguments | Opens a native file dialog (multi-select) |
| File path(s) | Process each file |
| Directory | Process all `.png` / `.jpg` / `.jpeg` / `.bmp` / `.webp` / `.tif` / `.tiff` inside |
| `--image PATH…` | Same as positional `images` (both can be combined) |

### Template

Provide a **tight crop** of one educational institute pin from a Google Maps screenshot (the colored disc + graduation-cap glyph + optional pin tip). Defaults to `template1.png` in the working directory.

Tips for a good template:

- Crop closely around a single pin; avoid map labels and roads.
- Prefer a sharp, unobscured icon at a typical zoom level.
- If icons look different at another zoom, either widen `--min-scale` / `--max-scale` or crop a second template.

---

## CLI reference

```text
python detect_icon.py [images ...] [options]
```

| Argument | Default | Description |
|---|---|---|
| `images` | *(file dialog)* | Screenshot file(s) or a folder |
| `--image PATH…` | `[]` | Same as positional images |
| `--template` | `template1.png` | Path to the cropped icon |
| `--threshold` | `0.80` | Minimum `TM_CCOEFF_NORMED` score for a candidate peak |
| `--min-scale` | `0.7` | Smallest template scale factor |
| `--max-scale` | `1.3` | Largest template scale factor |
| `--scale-step` | `0.05` | Scale increment between min and max |
| `--mode` | `auto` | `grayscale` \| `edges` \| `auto` |
| `--inner-threshold` | `0.88` | Minimum interior-glyph match; rejects other pin types |
| `--nms-iou` | `0.35` | IoU threshold for non-maximum suppression |
| `--output` | `output/detected.png` | Annotated output image path |

### Matching modes

| Mode | What it does |
|---|---|
| `grayscale` | Match on luminance only — fast, good baseline |
| `edges` | Gaussian blur + Canny edges — emphasizes shape over fill color |
| `auto` | Runs grayscale, edges, **and** color, then merges candidates |

`auto` is the recommended default for portfolio / production use.

---

## Output

### Console

For each screenshot the script prints:

1. A human-readable list of detections (index, box, center, confidence)
2. A JSON array of the same objects

Example detection object:

```json
{
  "x": 398,
  "y": 110,
  "width": 32,
  "height": 37,
  "center_x": 414,
  "center_y": 128,
  "confidence": 0.9932
}
```

| Field | Meaning |
|---|---|
| `x`, `y` | Top-left of the bounding box (pixels) |
| `width`, `height` | Box size at the matched scale |
| `center_x`, `center_y` | Pin center (useful for click / overlay placement) |
| `confidence` | Best multi-scale match score before / among fused modes |

If nothing matches:

```text
No matching educational institute icons found.
[]
```

### Annotated image

`draw_detections` overlays:

- Green rectangle around each pin
- Filled center circle
- Solid green tag with white text: `school`

Parent directories for `--output` are created automatically.

---

## How it works (technical)

Pipeline overview:

```text
Screenshot + Template
        │
        ▼
┌───────────────────────┐
│  Prepare pair(s)      │  grayscale / edges / color  (mode=auto → all)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Multi-scale match    │  resize template ∈ [min_scale, max_scale]
│  TM_CCOEFF_NORMED     │  3×3 local maxima ≥ threshold
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Interior glyph filter│  HSV saturation mask on pin disc
│  masked CCOEFF        │  drop pins that don’t share the glyph
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Non-max suppression  │  keep highest confidence; IoU < nms_iou
└───────────┬───────────┘
            ▼
     Boxes + annotated PNG
```

### 1. Image loading

`load_image` uses `cv2.imread(..., IMREAD_UNCHANGED)` and normalizes:

- 16-bit → 8-bit via `NORM_MINMAX`
- BGRA → BGR
- Rejects empty / unreadable files with clear errors

### 2. Multi-scale template matching

The screenshot stays at native resolution. The template is resized across:

```text
scale = min_scale, min_scale + step, …, max_scale
```

At each scale:

1. Skip if the scaled template is smaller than 8×8 or larger than the image.
2. Run `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`.
3. Keep **3×3 local maxima** at or above `--threshold` (dilate + equality mask), so plateaus don’t flood the candidate list.

Interpolation: `INTER_AREA` when shrinking, `INTER_CUBIC` when enlarging.

### 3. Why whole-pin matching isn’t enough

Google Maps pins share a similar teardrop silhouette. Matching the full pin shape alone tends to fire on **every** marker type (hospital, shopping, restaurant, etc.).

The important signal is the **inner disc + white glyph** (graduation cap vs bag vs cross). That is handled next.

### 4. Interior glyph filter

`filter_by_icon_interior`:

1. Build a mask of the colored pin interior from the template via HSV saturation (`S ≥ 35`), with morphological close/dilate. Fallback: intensity distance from border median if saturation is too weak.
2. For each candidate box, crop the screenshot patch and resize the template + mask to that size.
3. Compute a **masked** normalized cross-correlation (`_ccoeff_normed`) on gray pixels inside the mask only.
4. Keep the detection only if the score ≥ `--inner-threshold` (default **0.88**).

This step is what separates educational pins from other map icons that survived the outer shape match.

### 5. Non-maximum suppression

Overlapping boxes from different scales / modes are collapsed with greedy NMS:

- Sort by confidence descending
- Keep the best; discard any remaining box with IoU ≥ `--nms-iou` (default **0.35**)

### Key functions (API surface)

| Function | Role |
|---|---|
| `detect_educational_icons(...)` | End-to-end detection |
| `match_template_multiscale(...)` | Scale loop + local maxima |
| `filter_by_icon_interior(...)` | Glyph verification |
| `non_max_suppression(...)` | Overlap cleanup |
| `draw_detections(...)` | Visualization |
| `load_image(...)` | Safe image IO |

You can import these from another script without using the CLI:

```python
from detect_icon import load_image, detect_educational_icons, draw_detections
import cv2

image = load_image("screenshot.png")
template = load_image("template1.png")
dets = detect_educational_icons(image, template, mode="auto")
cv2.imwrite("out.png", draw_detections(image, dets))
```

---

## Tuning guide

| Symptom | Try this |
|---|---|
| Missed schools (false negatives) | Lower `--threshold` (e.g. `0.72–0.78`); widen scale range; use `--mode auto` |
| Extra non-school pins (false positives) | Raise `--inner-threshold` (e.g. `0.90–0.94`); raise `--threshold`; use a cleaner crop |
| Icons at very different zoom | Expand `--min-scale` / `--max-scale` (e.g. `0.5`–`1.6`) or provide a second template |
| Duplicate boxes on one pin | Lower `--nms-iou` slightly (more aggressive) or raise it if boxes are wrongly merged |
| Template larger than screenshot | Crop a smaller icon or lower `--min-scale` |
| Edge-heavy / low-color maps | Prefer `--mode edges` or keep `auto` |

### Sensible starting points

```bash
# Balanced (defaults)
python detect_icon.py ../test-images/test.png

# Strict (fewer false positives)
python detect_icon.py ../test-images/hospitaltest.png \
  --threshold 0.85 --inner-threshold 0.92

# Loose / multi-zoom
python detect_icon.py ../test-images/test4.png \
  --min-scale 0.55 --max-scale 1.5 --threshold 0.75
```

---

## Test images

Sample screenshots live in `test-images/`:

| File | Typical use |
|---|---|
| `test.png`, `test2.png`, `test3.png`, `test4.png` | Positive cases with one or more school pins |
| `input.png` | Additional map crop |
| `hospitaltest.png` | Control for non-school / mixed pins |

Run the suite as a folder:

```bash
python detect_icon.py ../test-images --output output/detected.png
```

---

## Limitations

- Designed for **Google Maps–style** educational pins that look like the provided template; other map styles may need a new crop.
- Classic template matching is sensitive to **heavy occlusion**, extreme blur, and dramatic style changes (dark mode, custom layers).
- Not a general object detector: it does not learn new icon classes without a new template and re-tuning.
- Confidence scores are correlation strengths, not calibrated probabilities.
- GUI picker requires a display / Tkinter; headless servers should pass paths explicitly.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All requested images processed (even if zero icons found) |
| `1` | One or more load / decode / write / validation errors |

---

## License & attribution

This portfolio project uses Google Maps screenshots as **test inputs**. Google Maps imagery and UI elements remain the property of Google LLC. Use your own screenshots for commercial work and respect Google’s terms of service.

---

## Author notes (portfolio)

Built as a **classical CV** alternative to YOLO / custom CNN pipelines when:

- You need a **fast, explainable** detector
- Training data is scarce
- The target icon is stable and can be cropped once
- Deploying on a laptop or simple backend without GPU is a requirement

Core idea: **match the pin, then verify the glyph** — silhouette alone is not discriminative on Google Maps.
