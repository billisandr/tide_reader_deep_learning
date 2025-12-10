# Models Directory

This directory contains pretrained models for water level detection.

## Directory Structure

```txt
models/
├── detection/          # Detection models (YOLO, RT-DETR, etc.)
│   └── best.pt        # Your trained detection model
└── segmentation/       # Segmentation models (SAM, etc.)
    └── sam_vit_h.pth  # SAM checkpoint (optional)
```

## Adding Models

### Option 1: Manual Upload

1. Place your trained detection model in `models/detection/`
2. Update the path in `config.yaml` under `models.detection.local_path`
3. If using SAM, place checkpoint in `models/segmentation/`

### Option 2: Download from Roboflow

1. Configure Roboflow credentials in `secrets.yaml`
2. Set model source to 'roboflow' in `config.yaml`
3. Specify workspace, project, and version
4. Model will be downloaded automatically on first run

## Model Formats

### Detection Models

- **YOLO**: `.pt` PyTorch format (YOLOv8 recommended)
- **RT-DETR**: `.pt` PyTorch format
- **Roboflow**: Automatically downloaded in correct format

### Segmentation Models (Optional)

- **SAM (Segment Anything Model)**: `.pth` checkpoint files
  - Download from: <https://github.com/facebookresearch/segment-anything#model-checkpoints>
  - Options: `vit_h` (default, 2.4GB), `vit_l` (1.2GB), `vit_b` (375MB)
  - Used for post-processing to refine waterline detection

#### SAM Waterline Refinement

When enabled, SAM refines the waterline detection through:

1. Segmenting the scale using the detected bounding box as a prompt
2. Analyzing color/hue changes within the scale region
3. Detecting the underwater portion of the scale (different color due to water)
4. Extracting the top boundary of the underwater region as the refined waterline

This provides more accurate waterline detection compared to using bounding box centers alone.

## Expected Model Output

Your detection model should detect the following classes:

- `waterline`: The water level line
- `scale`: The measurement scale
- `marker` (optional): Scale markers for calibration

Make sure your model is trained to detect these classes or update the `target_classes` in `config.yaml` accordingly.
