"""
Debug visualization module for tide_reader_DL.

Provides step-by-step visualization of the detection pipeline with configurable
output for debugging and analysis.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DebugVisualizer:
    """
    Manages debug visualization output for the water level detection pipeline.

    Creates timestamped session directories and saves annotated images at key
    processing stages.
    """

    def __init__(self, base_dir: str = "data/debug", enabled: bool = True):
        """
        Initialize the debug visualizer.

        Args:
            base_dir: Base directory for debug output
            enabled: Whether debug visualization is enabled
        """
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        self.session_dir: Optional[Path] = None
        self.step_count = 0
        self.saved_steps = []

        if self.enabled:
            self._create_session_directory()

    def _create_session_directory(self) -> None:
        """Create a timestamped session directory for this debug session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created debug session directory: {self.session_dir}")

    def save_debug_step(
        self,
        image: np.ndarray,
        step_name: str,
        description: str = ""
    ) -> None:
        """
        Save a debug image for a processing step.

        Args:
            image: Image to save (BGR format)
            step_name: Name of the step (used in filename)
            description: Optional description for logging
        """
        if not self.enabled or self.session_dir is None:
            return

        try:
            self.step_count += 1
            filename = f"{self.step_count:02d}_{step_name}.jpg"
            filepath = self.session_dir / filename

            cv2.imwrite(str(filepath), image)
            self.saved_steps.append(filename)

            log_msg = f"Saved debug step {self.step_count}: {step_name}"
            if description:
                log_msg += f" - {description}"
            logger.debug(log_msg)

        except Exception as e:
            logger.warning(f"Failed to save debug step '{step_name}': {e}")

    def annotate_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        title: str = "Detections",
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Annotate image with bounding boxes and labels for all detections.

        Args:
            image: Input image (BGR format)
            detections: List of detection dicts with keys: class, confidence, bbox
            title: Title to display at top of image
            show_confidence: Whether to show confidence scores

        Returns:
            Annotated image copy
        """
        annotated = image.copy()

        # Add title
        self._add_title(annotated, title)

        # Draw each detection
        for det in detections:
            class_name = det.get('class', 'unknown')
            confidence = det.get('confidence', 0.0)
            bbox = det.get('bbox')  # Expected format: [x1, y1, x2, y2]

            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Choose color based on class
            color = self._get_class_color(class_name)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = class_name
            if show_confidence:
                label += f" {confidence:.2f}"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated

    def annotate_waterline(
        self,
        image: np.ndarray,
        waterline_y: int,
        label: str = "Waterline",
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """
        Draw a horizontal line at the waterline position.

        Args:
            image: Input image (BGR format)
            waterline_y: Y-coordinate of waterline
            label: Label text for the waterline
            color: Line color in BGR

        Returns:
            Annotated image copy
        """
        annotated = image.copy()
        height, width = annotated.shape[:2]

        # Draw horizontal line
        cv2.line(annotated, (0, waterline_y), (width, waterline_y), color, 3)

        # Add label
        cv2.putText(
            annotated,
            label,
            (10, waterline_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        return annotated

    def annotate_sam_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        waterline_y: int,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay SAM segmentation mask on image with transparency.

        Args:
            image: Input image (BGR format)
            mask: Binary mask (0 or 1)
            waterline_y: Y-coordinate of refined waterline
            alpha: Transparency factor (0=transparent, 1=opaque)

        Returns:
            Image with mask overlay and waterline
        """
        annotated = image.copy()

        # Create colored mask overlay (blue for underwater region)
        mask_color = np.zeros_like(annotated)
        mask_color[mask > 0] = [255, 0, 0]  # Blue (BGR)

        # Blend mask with original image
        annotated = cv2.addWeighted(annotated, 1.0, mask_color, alpha, 0)

        # Draw refined waterline
        annotated = self.annotate_waterline(
            annotated,
            waterline_y,
            label="SAM Refined Waterline",
            color=(0, 255, 0)
        )

        return annotated

    def create_comparison_view(
        self,
        image: np.ndarray,
        bbox_waterline: int,
        sam_waterline: int
    ) -> np.ndarray:
        """
        Create side-by-side comparison of bbox vs SAM waterline detection.

        Args:
            image: Input image (BGR format)
            bbox_waterline: Y-coordinate from bounding box method
            sam_waterline: Y-coordinate from SAM refinement

        Returns:
            Comparison visualization
        """
        # Create two copies
        bbox_view = self.annotate_waterline(
            image.copy(),
            bbox_waterline,
            label="BBox Waterline",
            color=(0, 165, 255)  # Orange
        )

        sam_view = self.annotate_waterline(
            image.copy(),
            sam_waterline,
            label="SAM Waterline",
            color=(0, 255, 0)  # Green
        )

        # Combine side by side
        comparison = np.hstack([bbox_view, sam_view])

        # Add comparison text
        diff_pixels = abs(sam_waterline - bbox_waterline)
        comparison_text = f"Difference: {diff_pixels} pixels"

        cv2.putText(
            comparison,
            comparison_text,
            (comparison.shape[1] // 2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return comparison

    def annotate_measurements(
        self,
        image: np.ndarray,
        scale_top_y: int,
        waterline_y: int,
        pixels_per_cm: float,
        scale_above_water_cm: float,
        water_level_cm: float
    ) -> np.ndarray:
        """
        Annotate image with measurement breakdown.

        Args:
            image: Input image (BGR format)
            scale_top_y: Y-coordinate of scale top
            waterline_y: Y-coordinate of waterline
            pixels_per_cm: Calibration factor
            scale_above_water_cm: Calculated scale above water in cm
            water_level_cm: Calculated water level in cm

        Returns:
            Annotated image with measurement details
        """
        annotated = image.copy()
        height, width = annotated.shape[:2]

        # Draw scale top line
        cv2.line(annotated, (0, scale_top_y), (width, scale_top_y), (255, 0, 255), 2)
        cv2.putText(
            annotated,
            "Scale Top",
            (10, scale_top_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

        # Draw waterline
        cv2.line(annotated, (0, waterline_y), (width, waterline_y), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "Waterline",
            (10, waterline_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Draw dimension line on the right side
        dim_x = width - 50
        cv2.arrowedLine(
            annotated,
            (dim_x, scale_top_y),
            (dim_x, waterline_y),
            (0, 255, 0),
            2,
            tipLength=0.05
        )

        # Add pixel distance
        pixel_distance = waterline_y - scale_top_y
        mid_y = (scale_top_y + waterline_y) // 2
        cv2.putText(
            annotated,
            f"{pixel_distance}px",
            (dim_x + 10, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Add measurement info panel
        info_text = [
            f"Pixel distance: {pixel_distance} px",
            f"Calibration: {pixels_per_cm:.2f} px/cm",
            f"Scale above water: {scale_above_water_cm:.2f} cm",
            f"Water level: {water_level_cm:.2f} cm"
        ]

        self._add_info_panel(annotated, info_text, position='bottom')

        return annotated

    def add_final_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        waterline_y: int,
        water_level_cm: float,
        scale_above_water_cm: float,
        method: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Create final annotated result with all information.

        Args:
            image: Input image (BGR format)
            detections: Detection results
            waterline_y: Final waterline Y-coordinate
            water_level_cm: Calculated water level
            scale_above_water_cm: Scale measurement above water
            method: Detection method used (bbox/SAM/HSV)
            confidence: Detection confidence
            metadata: Additional metadata to display

        Returns:
            Final annotated image
        """
        annotated = image.copy()

        # Draw bounding boxes for detections
        for det in detections:
            bbox = det.get('bbox')
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            class_name = det.get('class', 'unknown')
            color = self._get_class_color(class_name)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Add class label
            cv2.putText(
                annotated,
                class_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Draw waterline
        annotated = self.annotate_waterline(annotated, waterline_y)

        # Add side panel with metadata
        side_panel_info = [
            f"Method: {method}",
            f"Confidence: {confidence:.3f}",
            f"",
            f"MEASUREMENTS:",
            f"Scale above water: {scale_above_water_cm:.2f} cm",
            f"Water level: {water_level_cm:.2f} cm",
        ]

        if metadata:
            side_panel_info.append("")
            side_panel_info.append("METADATA:")
            for key, value in metadata.items():
                side_panel_info.append(f"{key}: {value}")

        annotated = self._add_side_panel(annotated, side_panel_info)

        return annotated

    def _add_title(self, image: np.ndarray, title: str) -> None:
        """Add title text at the top of the image."""
        cv2.putText(
            image,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

    def _add_info_panel(
        self,
        image: np.ndarray,
        text_lines: List[str],
        position: str = 'bottom'
    ) -> None:
        """
        Add an information panel to the image.

        Args:
            image: Image to annotate (modified in place)
            text_lines: List of text lines to display
            position: 'top' or 'bottom'
        """
        height = image.shape[0]
        line_height = 25
        panel_height = len(text_lines) * line_height + 20

        # Create semi-transparent background
        overlay = image.copy()

        if position == 'bottom':
            y_start = height - panel_height
            y_end = height
        else:
            y_start = 0
            y_end = panel_height

        cv2.rectangle(overlay, (0, y_start), (image.shape[1], y_end), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Add text lines
        for i, line in enumerate(text_lines):
            y_pos = y_start + 20 + (i * line_height)
            cv2.putText(
                image,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    def _add_side_panel(
        self,
        image: np.ndarray,
        text_lines: List[str],
        panel_width: int = 300
    ) -> np.ndarray:
        """
        Add a side panel with information to the image.

        Args:
            image: Input image
            text_lines: List of text lines to display
            panel_width: Width of the side panel

        Returns:
            Image with side panel
        """
        height, width = image.shape[:2]

        # Create new image with extra width for panel
        result = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        result[:, :width] = image
        result[:, width:] = (40, 40, 40)  # Dark gray background

        # Add text to panel
        line_height = 25
        y_pos = 30

        for line in text_lines:
            cv2.putText(
                result,
                line,
                (width + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_pos += line_height

        return result

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a detection class."""
        colors = {
            'scale': (0, 255, 0),           # Green
            'underwater_scale': (255, 0, 0), # Blue
            'underwater': (255, 0, 0),       # Blue
            'default': (0, 165, 255)         # Orange
        }
        return colors.get(class_name.lower(), colors['default'])

    def cleanup_empty_session(self) -> None:
        """Remove session directory if no steps were saved."""
        if not self.enabled or self.session_dir is None:
            return

        if len(self.saved_steps) == 0:
            try:
                self.session_dir.rmdir()
                logger.info(f"Removed empty debug session: {self.session_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove empty session directory: {e}")

    def get_session_path(self) -> Optional[str]:
        """Get the path to the current debug session directory."""
        return str(self.session_dir) if self.session_dir else None
