# Quick Start Guide

Get the Water Level Detection System running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Windows/Linux/Mac
- A trained detection model or Roboflow account

## Step 1: Install Dependencies

```bash
# Activate virtual environment (if using one)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

## Step 2: Verify Setup

```bash
python verify_setup.py
```

This will check your installation and show what's missing.

## Step 3: Configure Model

Choose one option:

### Option A: Local Model

```bash
# Place your model file
cp your_model.pt models/detection/best.pt

# Edit config.yaml to use local model
# (Already configured by default)
```

### Option B: Roboflow API

```bash
# Copy secrets template
cp secrets.yaml.template secrets.yaml

# Edit secrets.yaml and add your API key
# Edit config.yaml:
#   source: 'roboflow-serverless'
#   roboflow_model_id: 'your-project/version'
```

## Step 4: Run Calibration

```bash
python -m src.calibration
```

A file browser will open:

1. Select an image with clear above-water scale markings
2. Click two points on the above-water scale
3. Enter the physical distance between the points (in cm)

The calibration data will be saved to config.yaml.

## Step 5: Download SAM Model (Optional)

SAM post-processing can refine waterline detection in some scenarios, but may not improve accuracy for all use cases.

**Note:** The DETR/YOLO detection model often provides accurate results on its own. Use the debug visualizer to compare bbox vs SAM waterline before committing to SAM.

To download the SAM segmentation model:

```bash
cd models/segmentation

# Choose ONE of these models:

# Option 1: vit_b (358 MB) - Fastest, good quality [RECOMMENDED]
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth

# Option 2: vit_l (1.2 GB) - Balanced
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O sam_vit_l.pth

# Option 3: vit_h (2.4 GB) - Best quality, slowest
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth
```

Then enable SAM in config.yaml:

```yaml
segmentation:
  enabled: true
  checkpoint_type: vit_b  # or vit_l, vit_h
  local_path: models/segmentation/sam_vit_b.pth
```

Skip this step to use only the detection model (faster and often equally accurate). Enable debug mode to visually compare detection methods.

## Step 6: Add Test Images

```bash
# Place your test images in:
data/input/
```

Or select a directory via GUI when running the system.

## Step 7: Run the System

```bash
python -m src.main
```

The system will:

1. Ask for configuration (input directory, SAM usage)
2. Load your model
3. Start processing images
4. Save results to database and export files

## Output

Results are saved in `data/output/`:

- `measurements_TIMESTAMP.csv` - CSV export
- `measurements_TIMESTAMP.json` - JSON export
- `measurements.db` - SQLite database
- `annotated/` - Annotated images showing detections

## Monitoring

Press `Ctrl+C` to stop the system gracefully.

## Troubleshooting

### "Calibration required" error

Run calibration:

```bash
python -m src.calibration
```

### "Model not found" error

Check your model path in config.yaml or verify Roboflow API key in secrets.yaml.

### "No module named 'X'" error

Install dependencies:

```bash
pip install -r requirements.txt
```

### "No images found" error

- Check that images are in the input directory
- Verify image extensions (.jpg, .jpeg, .png)
- Check DEBUG_MODE in .env (true = keeps originals, false = moves to processed)

## Configuration Tips

Edit [config.yaml](config.yaml):

```yaml
# Adjust detection sensitivity
models:
  detection:
    confidence_threshold: 0.90  # Lower = more detections, higher = more selective
    iou_threshold: 0.65

# Adjust processing
processing:
  save_annotated_images: true  # Set false to save disk space

# Adjust measurement
measurement:
  scale_height_cm: 45.0  # Update to match your physical scale
```

## What's Next

- Review results in `data/output/measurements.csv`
- Check annotated images in `data/output/annotated/`
- Adjust confidence thresholds if needed
- Set up automated monitoring (see README.md)

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) for calibration help
- See [STATUS.md](STATUS.md) for system status and setup verification
