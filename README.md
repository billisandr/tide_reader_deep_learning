# Water Level Detection System - Model-Based Approach

Automated water level detection using pretrained deep learning models (YOLO, RT-DETR) with optional SAM post-processing.

> Part of research conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [Technical University of Crete](https://www.tuc.gr/el/archi).

## Documentation Index

### Getting Started

- [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
- [CALIBRATION_GUIDE.md](docs/CALIBRATION_GUIDE.md) - Calibration instructions and best practices

### Roboflow Integration

- [ROBOFLOW_INFERENCE_GUIDE.md](docs/ROBOFLOW_INFERENCE_GUIDE.md) - Complete guide for Roboflow API usage
- [ROBOFLOW_SERVERLESS_GUIDE.md](docs/ROBOFLOW_SERVERLESS_GUIDE.md) - Hosted inference setup
- [ROBOFLOW_INFERENCE_CHANGES.md](docs/ROBOFLOW_INFERENCE_CHANGES.md) - Implementation changes log

### Project Information

- [STATUS.md](docs/STATUS.md) - Current system status and roadmap
- [CHANGELOG.md](CHANGELOG.md) - Version history and updates
- [LICENSE.md](LICENSE.md) - License information
- [models/README.md](models/README.md) - Model setup and download instructions

## Overview

This system uses pretrained object detection models to automatically measure water levels from images containing measurement scales. Unlike traditional computer vision approaches, it leverages deep learning for robust detection across varying conditions.

## Key Features

- Pretrained model support (YOLO, RT-DETR)
- Roboflow API integration for cloud models
- Optional SAM (Segment Anything) post-processing
- Interactive configuration at runtime
- Automated batch processing
- Multi-format data export (CSV, JSON, SQLite)
- GUI-based directory and model selection

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd tide_reader_comp_vision

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Model

***Option A: Local Model***

```bash
# Place your trained model in models/detection/
cp your_model.pt models/detection/best.pt

# Update config.yaml if using different path
```

***Option B: Roboflow Model (Download)***

```bash
# Copy secrets template
cp secrets.yaml.template secrets.yaml

# Edit secrets.yaml and add your Roboflow API key
# Update config.yaml with workspace/project/version
# Model will be downloaded on first run
```

***Option C: Roboflow Inference API (Hosted)***

```bash
# Copy secrets template
cp secrets.yaml.template secrets.yaml

# Edit secrets.yaml and add your Roboflow API key
# Update config.yaml:
#   source: 'roboflow-inference'
#   roboflow_model_id: 'workspace/project/version'
# No download required - uses hosted inference
```

See [ROBOFLOW_INFERENCE_GUIDE.md](ROBOFLOW_INFERENCE_GUIDE.md) for detailed instructions.

### 3. Run Calibration (Required)

Calibrate the pixels-per-cm ratio using an above-water scale reference:

```bash
python -m src.calibration
```

A file browser will open - select an image with clear above-water scale markings, then follow the on-screen instructions.

See [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) for detailed instructions.

**Why calibration is required:** The underwater portion of the scale is optically distorted by water refraction, so we must calibrate using only the above-water portion.

### 4. Configure Settings (Optional)

Edit [config.yaml](config.yaml) to adjust:

- Detection confidence thresholds
- Target detection classes
- Export formats

### 5. Run the System

```bash
python src/main.py
```

You'll be prompted to:

1. Select input directory (or use default)
2. Enable/disable SAM post-processing

## Configuration

### Model Configuration

The system expects your model to detect these classes:

- `waterline`: The water level line
- `scale`: The measurement scale
- `marker` (optional): Scale markers for calibration

Update `target_classes` in [config.yaml](config.yaml) if your model uses different class names.

### Measurement Calibration

**Auto-calibration**: If your model detects the scale, the system automatically calculates pixels-to-cm ratio.

**Manual calibration**: Set `pixels_per_cm` in [config.yaml](config.yaml) if you know the conversion factor.

### SAM Post-Processing

SAM can refine detections for:

- Precise scale boundary segmentation
- Underwater scale masking
- Improved waterline accuracy

Download SAM checkpoint from: [https://github.com/facebookresearch/segment-anything#model-checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

Place in `models/segmentation/sam_vit_h.pth`

## Project Structure

```txt
tide_reader_comp_vision/
├── config.yaml              # Main configuration
├── secrets.yaml            # API keys (create from template)
├── requirements.txt        # Python dependencies
├── src/
│   ├── main.py            # Main entry point
│   ├── detector.py        # Water level detection logic
│   ├── model_loader.py    # Model loading (local/Roboflow)
│   ├── database.py        # Database operations
│   └── utils.py           # Utility functions
├── models/
│   ├── detection/         # Place YOLO/RT-DETR models here
│   └── segmentation/      # Place SAM checkpoints here
└── data/
    ├── input/             # Input images
    ├── processed/         # Processed images
    └── output/            # Results (CSV, JSON, DB, annotated images)
```

## Workflow

1. **Startup**: User selects input directory, model, and SAM option
2. **Detection**: Model detects waterline and scale in images
3. **Measurement**: System calculates water level from detections
4. **Storage**: Results saved to database
5. **Export**: Data exported in configured formats (CSV/JSON/DB)
6. **Annotation**: Annotated images saved with bounding boxes and measurements

## Detection Process

```txt
Input Image
    ↓
[Detection Model (YOLO/RT-DETR/Roboflow)]
    ↓
Waterline + Scale Bounding Boxes
    ↓
[Optional: SAM Refinement]
    ↓
Precise Waterline Position
    ↓
[Measurement Calculation]
    ↓
Water Level (cm) + Scale Above Water (cm)
```

### SAM Waterline Refinement

When SAM post-processing is enabled, the system refines waterline detection through:

1. **Scale Segmentation**: Uses the detected scale bounding box as a prompt for SAM
2. **Color Analysis**: Analyzes HSV color space to detect underwater region
   - Underwater scales typically show color/hue differences due to water
   - Uses saturation gradient to find the transition point
3. **Boundary Extraction**: Identifies the top boundary of the underwater region
4. **Refined Waterline**: Uses this boundary as the precise waterline position

This provides significantly more accurate measurements compared to using bounding box centers, especially when:

- The waterline detection bbox is imprecise
- The scale shows clear color differentiation underwater
- High-precision measurements are required

To enable SAM refinement:

```yaml
# config.yaml
models:
  segmentation:
    enabled: true  # Enable SAM post-processing
    type: 'sam'
    source: 'local'
    local_path: 'models/segmentation/sam_vit_h.pth'
    checkpoint_type: 'vit_h'  # Options: vit_h, vit_l, vit_b
```

SAM requires downloading a checkpoint file (see [models/README.md](models/README.md)).

## Output Files

### Database

`data/output/measurements.db` - SQLite database with all measurements

### Exports (if enabled)

- `measurements_YYYYMMDD_HHMMSS.csv` - CSV format
- `measurements_YYYYMMDD_HHMMSS.json` - JSON format
- `measurements_YYYYMMDD_HHMMSS.db` - Database backup

### Annotated Images

`data/output/annotated/` - Images with detection overlays

Example filename: `annotated_success_20250829_143022_IMG_0001.jpg`

## Environment Variables

Create a `.env` file:

```bash
# Processing interval (seconds)
PROCESS_INTERVAL=60

# Debug mode (copies instead of moves processed images)
DEBUG_MODE=false

# GUI selector (force GUI mode)
USE_GUI_SELECTOR=false

# Database path
DB_PATH=data/output/measurements.db
```

## Training Your Own Model

To train a custom water level detection model:

1. **Collect images** of your measurement scale with various water levels
2. **Annotate images** with bounding boxes for:
   - Waterline (the water level line)
   - Scale (the measurement ruler/scale)
   - Markers (optional: scale markings)
3. **Train model** using:
   - [Roboflow](https://roboflow.com/) (recommended for beginners)
   - [Ultralytics YOLO](https://docs.ultralytics.com/modes/train/)
   - [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
4. **Export model** in PyTorch `.pt` format
5. **Place in** `models/detection/` directory

## Troubleshooting

### Model Not Loading

- Verify model file exists at specified path
- Check model format is `.pt` (PyTorch)
- Ensure model is compatible with ultralytics

### No Detections

- Check if target classes match your model's classes
- Lower `confidence_threshold` in config.yaml
- Verify images contain visible scale and waterline

### Inaccurate Measurements

- Enable SAM post-processing for better precision
- Adjust `pixels_per_cm` if manual calibration is used
- Ensure scale is clearly visible in images

### SAM Not Working

- Download correct SAM checkpoint (vit_h, vit_l, or vit_b)
- Place in `models/segmentation/`
- Check `checkpoint_type` in config.yaml matches file

## Advanced Usage

### Batch Processing

Place multiple images in input directory. System processes all unprocessed images automatically.

### Continuous Monitoring

System runs in a loop, checking for new images every `PROCESS_INTERVAL` seconds.

### Custom Classes

If your model uses different class names:

```yaml
processing:
  target_classes:
    - 'water_level'      # Instead of 'waterline'
    - 'ruler'            # Instead of 'scale'
```

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- Ultralytics (YOLO/RT-DETR)
- Segment Anything (optional)
- Roboflow (optional)
- OpenCV, NumPy, pandas

See [requirements.txt](requirements.txt) for complete list.

## License

BSD 3-Clause License. See [LICENSE.md](LICENSE.md) for details.

## Citation

If you use this system in your research, please cite the SenseLAB, Technical University of Crete.

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review [config.yaml](config.yaml) comments
3. Check model compatibility
4. Verify data directory structure
