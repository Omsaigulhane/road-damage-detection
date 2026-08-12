# AI-Based Smart Road Damage Detection and Severity Prediction System — Phase 1

A software MVP for detecting potholes and cracks in road images/video,
estimating their apparent and approximate physical size, and computing a
transparent 0–100 severity score. Built to run entirely on a laptop
(tested target: Windows, RTX 3050), with no paid cloud services and no
hardware dependency (camera-only pipeline; GPS/accelerometer are
Phase-2 extension points, not requirements).

**Status:** Phase 1 software foundation. See [Limitations](#limitations) —
this is not a validated measurement instrument.

---

## 1. Project Overview

The system takes a road image or video and:

1. Detects potholes and cracks with a fine-tuned YOLO model.
2. Draws bounding boxes with class + confidence.
3. Computes pixel-space size (width, height, area, relative area).
4. Optionally converts pixel size to an approximate physical size, if
   camera calibration parameters are provided — otherwise it clearly
   states the estimate is unavailable rather than guessing.
5. Computes a 0–100 severity score from a transparent, configurable
   weighted formula (not a learned/trained severity model).
6. Stores every detection in a local SQLite database.
7. Displays everything through a Streamlit dashboard, with a CLI that
   works independently of the dashboard.

## 2. Architecture

```
IMAGE / VIDEO -> INPUT PROCESSOR -> FRAME EXTRACTION -> PREPROCESSING
    -> YOLO DETECTOR -> [DETECTION DATA, BOUNDING BOX -> SIZE ESTIMATION]
    -> SEVERITY ENGINE -> 0-100 SEVERITY SCORE -> SQLite -> STREAMLIT DASHBOARD
```

Core inference logic (`src/pipeline.py`) is shared by both the CLI
(`scripts/detect.py`) and the dashboard (`dashboard/app.py`) — neither
depends on the other.

## 3. Requirements

- Python 3.10+
- Windows, macOS, or Linux
- NVIDIA GPU with CUDA strongly recommended for training (tested target:
  RTX 3050, 4GB VRAM); CPU works for inference and small-scale testing
- ~5GB free disk space for a converted dataset + trained weights

## 4. Installation

```bash
git clone <this-repo>
cd road_damage_detection
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### GPU (CUDA) setup on Windows with an RTX 3050

The default `torch` wheel in `requirements.txt` is CPU-only. For GPU
training, install the CUDA build instead (check your driver's supported
CUDA version first, e.g. via `nvidia-smi`):

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 5. Environment Setup

All tunables live in `config.yaml` — model choice, thresholds, severity
weights, dimension calibration, database path, output directories. Do not
edit source files to change these values.
## 6. Dataset Preparation

No dataset ships with this repository (Section 9 of the build spec — no
invented data). This project uses **RDD2020**
(Arya et al., "RDD2020: An annotated image dataset for automatic road
damage detection using deep learning", Data in Brief, 2021),
hosted on Mendeley Data: https://data.mendeley.com/datasets/5ty2wb6gvg/1

RDD2020 was used instead of the newer RDD2022 because RDD2022's
per-country FigShare download links returned `AccessDenied` errors at the
time of this project (the only working FigShare link bundled all six
countries into a single 12.35 GB archive, impractical given the project
deadline). RDD2020 is from the same research group, uses the identical
PASCAL VOC XML annotation format, and covers the same four damage
categories (D00/D10/D20/D40), so it is a direct, equivalent substitute —
just smaller (India/Japan/Czech only, ~1.1 GB for the labelled train set).

This project specifically uses the **Czech Republic** subset of RDD2020's
`train.tar.gz`.

**Download manually** (not automated here — distributed as a Mendeley
archive, outside this project's dependency scope):

1. Visit https://data.mendeley.com/datasets/5ty2wb6gvg/1
2. Download `train.tar.gz` (1.37 GB — this is the only file needed;
   `test1.tar.gz`/`test2.tar.gz` are images without annotations)
3. Extract it (e.g. with 7-Zip or WinRAR — two-step extraction since it's
   a `.tar.gz`), giving a folder structure like:
   `train/train/Czech/images/`, `train/train/Czech/annotations/xmls/`
   (also contains `India/` and `Japan/` subfolders if you want to add
   more training data later)

**Convert to this project's 2-class YOLO format** (merges D00/D10/D20 →
`crack`, D40 → `pothole`; other categories are out of scope and skipped):

```bash
python scripts/prepare_dataset.py --raw-dir /path/to/train/train/Czech --output-dir data/processed
```

Repeat per country and re-run, or extend the script to merge multiple
countries in one pass if you want more training data — the script reports
real, measured counts to `outputs/reports/dataset_preparation_report.json`;
nothing is invented. In this project's actual run: 1,072 images converted
(1,745 annotations: 197 pothole, 1,548 crack), split 803/161/108
(train/val/test).

**Validate before training:**

```bash
python scripts/validate_dataset.py --dataset-yaml data/dataset.yaml
```

This checks for missing images/labels, malformed labels, invalid class
IDs, out-of-range boxes, empty annotation files, corrupted images, and
duplicates, and writes `outputs/reports/dataset_validation_report.json`.
Training refuses to start if critical issues are found.
## 7. Training

```bash
python scripts/train.py --model yolo26n.pt --data data/dataset.yaml \
    --epochs 100 --batch 16 --imgsz 640 --device auto
```

- Fine-tunes from pretrained COCO weights (never trains from scratch).
- Auto-detects CUDA; falls back to CPU with a warning if unavailable.
- On a 4GB-class GPU (RTX 3050), reduce `--batch` to 8 or 4 if you hit
  CUDA out-of-memory errors.
- Outputs (best/last weights, metrics, plots, confusion matrix) are
  saved under `models/trained/run/`.
- To benchmark the fallback model instead: `--model yolo11n.pt`.

After training, point `config.yaml -> model.trained_weights` at
`models/trained/run/weights/best.pt`.

## 8. Evaluation

```bash
python scripts/evaluate.py --weights models/trained/run/weights/best.pt --data data/dataset.yaml --split test
```

Reports precision, recall, F1, mAP@0.5, mAP@0.5:0.95, class-wise
precision/recall/AP, inference latency, FPS, parameter count, and model
size — all measured on your actual run, written to
`outputs/reports/evaluation_report.json`. No target numbers are
substituted for unmeasured values; unmeasured fields read
`"Not measured yet."`.

## 9. Image Inference

```bash
python scripts/detect.py --source path/to/image.jpg
```

Loads the model, runs detection, filters by the configured confidence
threshold, draws boxes with class/confidence/severity, saves the
annotated image to `outputs/images/`, and stores results in SQLite
(disable with `--no-save` / `--no-store`).

## 10. Video Inference

```bash
python scripts/detect.py --source path/to/video.mp4
```

Samples frames per `video.frame_sample_rate` in `config.yaml` (default:
every 5th frame), runs the same detection pipeline per sampled frame, and
writes an annotated `.mp4` to `outputs/videos/`. Full multi-object
tracking is intentionally not implemented in Phase 1; an optional simple
IoU + temporal-proximity event aggregation can be turned on via
`video.event_aggregation.enabled` in `config.yaml` to reduce trivial
frame-to-frame duplicate counting.

## 11. Dashboard

```bash
streamlit run dashboard/app.py
```

Provides: summary metrics, image/video upload, annotated result display,
detection details table, history table, and basic charts (potholes vs.
cracks, severity distribution).

## 12. Configuration

See `config.yaml` for every tunable value: model name/path, confidence/IoU
thresholds, image size, video frame sampling, database path, output
directories, severity weights/thresholds, and dimension-estimation
calibration parameters. Nothing is hard-coded in source.

## 13. Severity Methodology

The severity score is a **transparent, rule/weight-based engineering
score**, not a scientifically validated physical damage assessment and
not a trained model (no severity-labelled ground truth currently exists
for this project):

```
size_score       = clamp(relative_area / size_reference, 0, 1)
dimension_score  = clamp(estimated_size_m / dimension_reference, 0, 1)
                    (falls back to size_score if no calibration is available)
confidence_score = YOLO confidence (0-1), used as a reliability weight,
                    not a direct severity proxy

severity_raw   = W_SIZE * size_score + W_DIMENSION * dimension_score
                 + W_CONFIDENCE * confidence_score
severity_score = clamp(severity_raw * 100, 0, 100)
```

Default weights (`config.yaml -> severity.weights`) and category
thresholds (`config.yaml -> severity.thresholds`) are initial engineering
values, not externally validated standards, and are meant to be tuned.

## 14. Limitations

1. No physical sensor data (accelerometer/GPS) in Phase 1.
2. No GPS in Phase 1 — location fields exist in the schema but are unused.
3. No calibrated depth measurement — dimension estimation uses a flat-
   ground pinhole-camera approximation that requires manual calibration
   (`camera_height_m`, `focal_length_px`); without it, only pixel
   measurements are reported.
4. Physical dimensions are therefore approximate, not precise.
5. Severity score is rule/weight based, not learned from severity-labelled
   data.
6. Lighting and weather conditions affect visual detection quality.
7. Camera perspective/angle affects size estimation accuracy.
8. Training data is limited to the Czech Republic subset of RDD2020
   (1,072 images); the model has not been validated on road/damage
   imagery from other countries or road types, and generalization is
   unknown.
9. Training data has significant class imbalance (141 pothole vs. 1,137
   crack annotations), which measurably reduced pothole detection
   performance relative to crack detection on the held-out test set
   (pothole mAP50: 0.182 vs. crack mAP50: 0.386; see
   `outputs/reports/evaluation_report.json`).
10. Overall detection accuracy (mAP50: 0.284, precision: 0.365, recall:
    0.352 on the test set) reflects the small training set size (803
    images) and limited training time; this is a Phase-1 baseline, not
    a production-ready detector.
11. No real vehicle deployment has been performed in Phase 1.

## 15. Future Hardware Integration (Phase 2)

`src/sensors/base.py` defines `VibrationSensorProvider` and
`GPSSensorProvider` interfaces with Phase-1 `Null*` implementations.
`src/sensors/future_mpu6050.py` documents (but does not implement)
`MPU6050Provider` and `GPSNeo6MProvider` — Phase 2 can wire real hardware
into the severity engine and database's existing (currently NULL)
`vibration_magnitude`, `latitude`, `longitude` fields without touching the
rest of the application.

## 16. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ConfigError: Configuration file not found` | Run scripts from the project root, or pass `--dataset-yaml` / equivalent explicitly. |
| `DetectorError: CUDA was requested but no CUDA-capable GPU was detected` | Set `model.device: auto` or `cpu` in `config.yaml`, or install the CUDA build of `torch`. |
| `RuntimeError: CUDA out of memory` during training | Lower `--batch` (try 8, then 4) or `--imgsz`. |
| `ImageProcessingError: Failed to decode image` | File is corrupted or not a supported format (`.jpg .jpeg .png .bmp .webp`). |
| `VideoProcessingError: Failed to open video` | File is corrupted or not a supported format (`.mp4 .avi .mov .mkv`). |
| Dimension estimate shows "Unavailable" | Expected — set `camera_height_m` and `focal_length_px` in `config.yaml` under `dimension_estimation` to enable approximate physical sizing. |
| `scripts/train.py` exits before training starts | Dataset validation failed — check `outputs/reports/dataset_validation_report.json` and fix the reported issues. |

## 17. Project Structure

```
road_damage_detection/
├── config.yaml
├── requirements.txt
├── data/                  (raw/, processed/, dataset.yaml, road_damage.db)
├── models/                (pretrained/, trained/)
├── src/
│   ├── pipeline.py         (shared inference pipeline)
│   ├── detection/          (detector.py, postprocess.py)
│   ├── preprocessing/      (image_processor.py)
│   ├── dimensions/         (estimator.py)
│   ├── severity/           (severity_engine.py)
│   ├── video/               (video_processor.py)
│   ├── database/            (database.py, models.py, repository.py)
│   ├── sensors/              (base.py, future_mpu6050.py)
│   └── utils/                (config.py, logger.py)
├── scripts/                 (prepare_dataset.py, validate_dataset.py, train.py, evaluate.py, detect.py)
├── dashboard/                (app.py)
├── tests/
└── outputs/                  (images/, videos/, reports/)
```

## 18. Testing

```bash
pytest tests/ -v
```

Covers severity bounds/edge-cases, dimension-estimator calibration
handling, bounding-box metrics, detector parsing/error-handling (mocked
model), database CRUD/validation, and video error handling. All 30 tests
pass in this environment (validated without GPU; training itself must be
run on a CUDA-capable machine).
