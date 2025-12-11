# Scale Calibration Guide

## Why Calibration is Required

The underwater portion of the scale is **optically distorted** due to water refraction. This means we cannot use the total scale height to calculate the pixels-per-cm ratio.

Instead, we must calibrate using **only the above-water portion** of the scale, where there is no optical distortion.

## Calibration Process

### Step 1: Prepare Calibration Image

Take a photo of your scale setup where:
- The **above-water portion is clearly visible**
- Scale markings are readable
- The image is well-lit
- You know the physical distance between two visible markings

Example: If your scale has markings every 5cm, you can use two consecutive markings.

### Step 2: Run Calibration Script

Simply run:
```bash
python -m src.calibration
```

A file browser will open automatically. Navigate to and select your calibration image.

**Alternative:** You can also provide the path directly:
```bash
python -m src.calibration path/to/calibration_image.jpg
```

### Step 3: Select Two Points

A window will open showing your image:

1. **Click on the FIRST point** (e.g., top of above-water scale or a marking)
2. **Click on the SECOND point** (e.g., bottom of above-water scale or another marking)
3. **Press any key** to continue

**IMPORTANT:** Only select points on the **above-water** portion of the scale!

### Step 4: Enter Physical Distance

The script will display the pixel distance and ask:

```
Enter the physical distance between these two points (in cm):
```

Enter the actual physical distance between the two points you clicked.

**Example:**
- If you clicked on the 40cm and 30cm markings, enter: `10`
- If you clicked on the 45cm and 20cm markings, enter: `25`

### Step 5: Verification

The calibration data will be saved to `config.yaml`:

```yaml
measurement:
  pixels_per_cm: 8.75  # Example value

calibration:
  pixels_per_cm: 8.75
  calibration_image: /absolute/path/to/image.jpg
  image_width: 1920
  image_height: 1080
  point1: [450, 200]
  point2: [450, 875]
  pixel_distance: 675.0
  physical_distance_cm: 77.0
```

## Important Notes

### Do NOT Use Total Scale Height

The total scale includes both above-water and underwater portions. The underwater portion is distorted by refraction, which would give an **incorrect** pixels-per-cm ratio.

**Wrong approach:**
```
Total scale: 100cm
Scale height in image: 850 pixels
pixels_per_cm = 850 / 100 = 8.5  # WRONG - includes distorted underwater portion
```

**Correct approach:**
```
Above-water portion: 25cm (measured between two visible markings)
Distance in image: 218 pixels
pixels_per_cm = 218 / 25 = 8.72  # CORRECT - no distortion
```

### Handling Image Resizing

The calibration script automatically handles:
- Roboflow's image resizing (e.g., to 432x432)
- Display scaling for large images
- Coordinate mapping back to original image space

You don't need to worry about these - just use your original high-resolution image for calibration.

### When to Re-calibrate

Re-run calibration if:
- Camera position changes
- Camera zoom/focus changes
- Scale position changes
- You switch to a different camera
- Image resolution changes

## Troubleshooting

### Error: "Calibration required!"

```
ValueError: Calibration required! Run calibration:
  python -m src.calibration <calibration_image_path>
```

**Solution:** You haven't run calibration yet. Follow Step 2 above.

### Error: "pixels_per_cm not calibrated"

The `config.yaml` has `pixels_per_cm: null`. Run the calibration script to set this value.

### Calibration seems inaccurate

**Possible causes:**
1. **Used underwater portion**: Make sure both points are above water
2. **Wrong physical distance**: Double-check the distance you entered
3. **Scale markings unclear**: Use a clearer image
4. **Camera angle changed**: Re-calibrate with current camera position

### Display window too small/large

The calibration script automatically scales the display window. The coordinates are automatically mapped back to the original image, so the display size doesn't affect accuracy.

## Example Workflow

```bash
# 1. Take calibration photo (scale clearly visible above water)
# Save as: data/input/calibration.jpg

# 2. Run calibration
python -m src.calibration data/input/calibration.jpg

# 3. Click two points on above-water scale markings
# (Window appears, click two points, press any key)

# 4. Enter the physical distance
# Prompt: "Enter the physical distance between these two points (in cm): "
# You type: 20

# 5. Calibration saved!
# Output: "Calibration saved! pixels_per_cm: 8.75"

# 6. Run the main system
python src/main.py
```

## Verification

After calibration, run a test image and check the logs:

```
INFO: Using calibrated pixels_per_cm: 8.75
INFO: Scale above water: 218.0 pixels = 24.9 cm
INFO: Water level: 20.1 cm
```

Verify that the measurements match your expected values based on the physical scale.
