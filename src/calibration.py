"""
Calibration module for establishing pixels-per-cm ratio using above-water scale.
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional


class ScaleCalibrator:
    """Handles scale calibration to establish accurate pixels-per-cm ratio."""

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize calibrator.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)

    def calibrate_interactive(self, image_path: str) -> Dict[str, float]:
        """
        Interactive calibration using a reference image.

        User clicks two points on the above-water scale and provides the
        physical distance between them in cm.

        Args:
            image_path: Path to calibration image

        Returns:
            Dictionary with calibration results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read calibration image: {image_path}")

        orig_height, orig_width = image.shape[:2]
        self.logger.info(f"Calibration image: {orig_width}x{orig_height}")

        # Resize for display if too large
        display_image = image.copy()
        scale_factor = 1.0

        max_display_height = 800
        if orig_height > max_display_height:
            scale_factor = max_display_height / orig_height
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)
            display_image = cv2.resize(display_image, (new_width, new_height))
            self.logger.info(f"Display scaled to: {new_width}x{new_height} (factor: {scale_factor:.3f})")

        # Collect two points from user
        points = []
        window_name = "Calibration - Click two points on above-water scale"

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Scale back to original coordinates
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                points.append((orig_x, orig_y))

                # Draw on display image
                cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
                if len(points) == 2:
                    cv2.line(display_image, (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor)),
                            (int(points[1][0] * scale_factor), int(points[1][1] * scale_factor)),
                            (0, 255, 0), 2)
                cv2.imshow(window_name, display_image)

                self.logger.info(f"Point {len(points)}: ({orig_x}, {orig_y})")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, display_image)

        print("\n=== Scale Calibration ===")
        print("Instructions:")
        print("1. Click on the TOP of the above-water scale (e.g., top of visible scale)")
        print("2. Click on the BOTTOM of the above-water scale (e.g., near waterline)")
        print("3. Press any key to continue")
        print("\nNote: Only use the ABOVE-WATER portion to avoid refraction distortion!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(points) != 2:
            raise ValueError("Need exactly 2 points for calibration")

        # Calculate pixel distance
        pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)

        # Get physical distance from user
        print(f"\nPixel distance between points: {pixel_distance:.1f} pixels")
        physical_distance_cm = float(input("Enter the physical distance between these two points (in cm): "))

        # Calculate pixels per cm
        pixels_per_cm = pixel_distance / physical_distance_cm

        calibration_data = {
            'pixels_per_cm': pixels_per_cm,
            'calibration_image': str(Path(image_path).absolute()),
            'image_width': orig_width,
            'image_height': orig_height,
            'point1': points[0],
            'point2': points[1],
            'pixel_distance': float(pixel_distance),
            'physical_distance_cm': physical_distance_cm
        }

        self.logger.info(f"Calibration complete: {pixels_per_cm:.2f} pixels/cm")

        return calibration_data

    def save_calibration(self, calibration_data: Dict[str, float]):
        """
        Save calibration data to config file.

        Args:
            calibration_data: Calibration results
        """
        # Load existing config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Convert numpy types to native Python types
        clean_data = {}
        for key, value in calibration_data.items():
            if isinstance(value, np.ndarray):
                clean_data[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                clean_data[key] = float(value)
            elif isinstance(value, tuple):
                clean_data[key] = list(value)
            else:
                clean_data[key] = value

        # Update calibration section
        config['measurement']['pixels_per_cm'] = float(clean_data['pixels_per_cm'])

        if 'calibration' not in config:
            config['calibration'] = {}

        config['calibration'].update(clean_data)

        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Calibration saved to {self.config_path}")
        print(f"\nCalibration saved!")
        print(f"  pixels_per_cm: {calibration_data['pixels_per_cm']:.2f}")
        print(f"  Reference: {calibration_data['physical_distance_cm']} cm = {calibration_data['pixel_distance']:.1f} pixels")


def select_calibration_image() -> Optional[str]:
    """
    Open file browser to select calibration image.

    Returns:
        Selected image path or None if cancelled
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    print("\n=== Water Level Calibration Tool ===")
    print("Opening file browser to select calibration image...")
    print("\nSelect an image with ABOVE-WATER scale markings clearly visible.")

    file_path = filedialog.askopenfilename(
        title="Select Calibration Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ],
        initialdir="data/input"
    )

    root.destroy()

    if not file_path:
        print("No image selected. Calibration cancelled.")
        return None

    print(f"Selected: {file_path}\n")
    return file_path


def main():
    """Run calibration from command line."""
    import sys

    logging.basicConfig(level=logging.INFO)

    # Check if image path provided as argument (backwards compatibility)
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        print(f"\n=== Water Level Calibration Tool ===")
        print(f"Using provided image: {image_path}\n")
    else:
        # Use GUI file browser
        image_path = select_calibration_image()
        if image_path is None:
            sys.exit(1)

    calibrator = ScaleCalibrator()

    try:
        calibration_data = calibrator.calibrate_interactive(image_path)
        calibrator.save_calibration(calibration_data)
        print("\n=== Calibration Complete ===")
        print("You can now run the main system: python src/main.py")
    except Exception as e:
        print(f"\nCalibration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
