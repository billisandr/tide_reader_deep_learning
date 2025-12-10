"""
Model-based water level detector using YOLO/RT-DETR and optional SAM post-processing.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import supervision as sv


class WaterLevelDetector:
    """Detects water levels using pretrained detection models."""

    def __init__(self, config: Dict[str, Any], detection_model, segmentation_model=None):
        """
        Initialize the detector.

        Args:
            config: Configuration dictionary
            detection_model: Loaded detection model (YOLO/RT-DETR/Roboflow)
            segmentation_model: Optional SAM model for post-processing
        """
        self.config = config
        self.detection_model = detection_model
        self.segmentation_model = segmentation_model
        self.logger = logging.getLogger(__name__)

        # Detection parameters
        self.conf_threshold = config['models']['detection']['confidence_threshold']
        self.iou_threshold = config['models']['detection']['iou_threshold']
        self.target_classes = config['processing']['target_classes']

        # Measurement parameters
        self.pixels_per_cm = config['measurement'].get('pixels_per_cm')
        self.scale_height_cm = config['measurement']['scale_height_cm']
        self.measurement_method = config['measurement']['method']

        # Output directories
        self.output_dir = Path('data/output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config['processing']['save_annotated_images']:
            self.annotated_dir = self.output_dir / 'annotated'
            self.annotated_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Detector initialized with confidence={self.conf_threshold}")
        self.logger.info(f"Target classes: {self.target_classes}")

    def process_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single image to detect water level.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with detection results or None if failed
        """
        self.logger.info(f"Processing image: {Path(image_path).name}")

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None

            # Run detection
            detections = self._run_detection(image)

            if detections is None or len(detections) == 0:
                self.logger.warning(f"No detections found in {Path(image_path).name}")
                return self._create_failed_result(image_path, image, "No detections")

            # Extract waterline and scale information
            result = self._extract_measurements(image, detections, image_path)

            # Save annotated image if enabled
            if self.config['processing']['save_annotated_images']:
                self._save_annotated_image(image, detections, result, image_path)

            return result

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            return None

    def _run_detection(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run detection model on image."""
        try:
            # Check model type and run inference
            model_type = self.config['models']['detection']['type']
            source = self.config['models']['detection']['source']

            # Check source first - Roboflow has different API
            if source == 'roboflow-serverless':
                # Roboflow Serverless API (inference-sdk)
                import cv2
                import tempfile
                import os

                # Save image to temp file (inference-sdk requires file path)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, image)
                    tmp_path = tmp.name

                try:
                    client = self.detection_model['client']
                    model_id = self.detection_model['model_id']

                    results = client.infer(tmp_path, model_id=model_id)
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)

                if not results or 'predictions' not in results or len(results['predictions']) == 0:
                    return None

                # Get image dimensions for coordinate scaling
                # Roboflow resizes to model input size (e.g., 432x432) but we need original coordinates
                orig_height, orig_width = image.shape[:2]

                # Get inference dimensions from response (Roboflow includes this)
                inference_width = results.get('image', {}).get('width', orig_width)
                inference_height = results.get('image', {}).get('height', orig_height)

                # Calculate scaling factors
                scale_x = orig_width / inference_width
                scale_y = orig_height / inference_height

                self.logger.debug(f"Original image: {orig_width}x{orig_height}, "
                                f"Inference size: {inference_width}x{inference_height}, "
                                f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

                boxes = []
                confidences = []
                class_names = []

                for pred in results['predictions']:
                    # Get bbox in inference space (432x432)
                    x_center = pred['x']
                    y_center = pred['y']
                    width = pred['width']
                    height = pred['height']

                    # Convert to xyxy format in inference space
                    x1_inf = x_center - width/2
                    y1_inf = y_center - height/2
                    x2_inf = x_center + width/2
                    y2_inf = y_center + height/2

                    # Scale to original image space
                    x1 = x1_inf * scale_x
                    y1 = y1_inf * scale_y
                    x2 = x2_inf * scale_x
                    y2 = y2_inf * scale_y

                    boxes.append([x1, y1, x2, y2])
                    confidences.append(pred['confidence'])
                    class_names.append(pred['class'])

                detections = {
                    'boxes': np.array(boxes),
                    'confidences': np.array(confidences),
                    'class_names': class_names
                }

                self.logger.debug(f"Roboflow Serverless returned {len(boxes)} detections (scaled to original image)")

            elif source in ['roboflow', 'roboflow-inference']:
                # Roboflow model or Roboflow Inference API
                results = self.detection_model.predict(image, confidence=int(self.conf_threshold * 100))

                if not results or len(results.predictions) == 0:
                    return None

                boxes = []
                confidences = []
                class_names = []

                for pred in results.predictions:
                    boxes.append([pred.x - pred.width/2, pred.y - pred.height/2,
                                  pred.x + pred.width/2, pred.y + pred.height/2])
                    confidences.append(pred.confidence)
                    class_names.append(pred.class_name)

                detections = {
                    'boxes': np.array(boxes),
                    'confidences': np.array(confidences),
                    'class_names': class_names
                }

                self.logger.debug(f"Roboflow returned {len(boxes)} detections")

            elif model_type in ['yolo', 'rt-detr'] and source == 'local':
                # Ultralytics models (local)
                results = self.detection_model.predict(
                    image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )

                if len(results) == 0 or len(results[0].boxes) == 0:
                    return None

                # Extract detections
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                class_names = [result.names[int(cls)] for cls in class_ids]

                detections = {
                    'boxes': boxes,
                    'confidences': confidences,
                    'class_ids': class_ids,
                    'class_names': class_names
                }

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.logger.info(f"Detected {len(detections['boxes'])} objects")
            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}", exc_info=True)
            return None

    def _refine_waterline_with_sam(self, image: np.ndarray, scale_box: np.ndarray) -> Optional[float]:
        """
        Refine waterline detection using SAM to segment underwater scale region.

        Args:
            image: Input image
            scale_box: Bounding box of the scale [x1, y1, x2, y2]

        Returns:
            Refined waterline Y-coordinate or None if refinement fails
        """
        if self.segmentation_model is None:
            return None

        try:
            self.logger.info("Refining waterline with SAM segmentation")

            # Convert image to RGB (SAM expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Set image for SAM predictor
            from segment_anything import SamPredictor
            predictor = SamPredictor(self.segmentation_model)
            predictor.set_image(image_rgb)

            # Use scale bounding box as prompt for SAM
            input_box = np.array(scale_box, dtype=np.float32)

            # Generate segmentation mask
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,
            )

            # Select best mask (highest score)
            best_mask_idx = np.argmax(scores)
            scale_mask = masks[best_mask_idx]

            self.logger.debug(f"SAM segmentation score: {scores[best_mask_idx]:.3f}")

            # Extract underwater region from the mask
            # The underwater region is the lower portion of the scale
            # We'll detect the color/hue boundary in the masked region
            underwater_mask = self._detect_underwater_region(image_rgb, scale_mask, scale_box)

            if underwater_mask is None:
                self.logger.warning("Failed to detect underwater region")
                return None

            # Find the top boundary of the underwater region (waterline)
            waterline_y = self._extract_waterline_from_mask(underwater_mask)

            if waterline_y is not None:
                self.logger.info(f"SAM refined waterline at y={waterline_y:.1f}")

            return waterline_y

        except Exception as e:
            self.logger.error(f"SAM refinement failed: {e}", exc_info=True)
            return None

    def _detect_underwater_region(self, image_rgb: np.ndarray, scale_mask: np.ndarray,
                                  scale_box: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect underwater region of scale using color/hue analysis.

        Args:
            image_rgb: RGB image
            scale_mask: Binary mask of the scale region
            scale_box: Bounding box of the scale

        Returns:
            Binary mask of underwater region or None if detection fails
        """
        try:
            # Convert to HSV for better color analysis
            image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

            # Extract scale region using bounding box
            x1, y1, x2, y2 = scale_box.astype(int)
            scale_region_hsv = image_hsv[y1:y2, x1:x2]
            scale_region_mask = scale_mask[y1:y2, x1:x2]

            # Get hue and saturation values within the scale mask
            hue = scale_region_hsv[:, :, 0][scale_region_mask]
            saturation = scale_region_hsv[:, :, 1][scale_region_mask]
            value = scale_region_hsv[:, :, 2][scale_region_mask]

            if len(hue) == 0:
                return None

            # Analyze vertical gradient to find color transition
            height, width = scale_region_hsv.shape[:2]
            vertical_profile = np.zeros(height)

            for y in range(height):
                row_mask = scale_region_mask[y, :]
                if np.sum(row_mask) > 0:
                    # Average hue and saturation for this row
                    row_hue = scale_region_hsv[y, :, 0][row_mask]
                    row_sat = scale_region_hsv[y, :, 1][row_mask]
                    vertical_profile[y] = np.mean(row_sat)  # Use saturation as indicator

            # Find the transition point (waterline)
            # Look for significant change in saturation (underwater typically has different color)
            if len(vertical_profile) < 10:
                return None

            # Smooth the profile
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(vertical_profile, sigma=3)

            # Find gradient (rate of change)
            gradient = np.gradient(smoothed)

            # Find the point with maximum positive gradient (transition to underwater)
            # Look in the lower half of the scale
            search_start = height // 3
            search_region = gradient[search_start:]

            if len(search_region) == 0:
                return None

            # Find peak gradient
            transition_idx = search_start + np.argmax(np.abs(search_region))

            # Create underwater mask (everything below transition point)
            underwater_mask = np.zeros_like(scale_mask, dtype=np.uint8)
            underwater_mask[y1 + transition_idx:y2, x1:x2] = scale_region_mask[transition_idx:, :]

            self.logger.debug(f"Detected underwater transition at relative y={transition_idx}")

            return underwater_mask

        except Exception as e:
            self.logger.error(f"Underwater region detection failed: {e}", exc_info=True)
            return None

    def _extract_waterline_from_mask(self, underwater_mask: np.ndarray) -> Optional[float]:
        """
        Extract waterline Y-coordinate from underwater mask.

        Args:
            underwater_mask: Binary mask of underwater region

        Returns:
            Waterline Y-coordinate or None if extraction fails
        """
        try:
            # Find all non-zero points in mask
            points = np.where(underwater_mask > 0)

            if len(points[0]) == 0:
                return None

            # Waterline is the minimum Y coordinate (top boundary)
            waterline_y = float(np.min(points[0]))

            return waterline_y

        except Exception as e:
            self.logger.error(f"Waterline extraction failed: {e}", exc_info=True)
            return None

    def _extract_measurements(self, image: np.ndarray, detections: Dict[str, Any],
                            image_path: str) -> Dict[str, Any]:
        """
        Extract water level measurements from detections.

        Args:
            image: Input image
            detections: Detection results
            image_path: Path to image file

        Returns:
            Measurement results dictionary
        """
        boxes = detections['boxes']
        class_names = detections['class_names']
        confidences = detections['confidences']

        # Find scale and underwater_scale detections
        scale_boxes = []
        underwater_scale_boxes = []

        for i, class_name in enumerate(class_names):
            if class_name.lower() in ['scale', 'ruler', 'measurement']:
                scale_boxes.append((boxes[i], confidences[i]))
            elif class_name.lower() in ['underwater_scale', 'underwater-scale', 'underwater scale']:
                underwater_scale_boxes.append((boxes[i], confidences[i]))

        # Check if we have the required detections
        if not scale_boxes:
            self.logger.warning("No scale detected")
            return self._create_failed_result(image_path, image, "No scale detected")

        if not underwater_scale_boxes:
            self.logger.warning("No underwater_scale detected")
            return self._create_failed_result(image_path, image, "No underwater_scale detected")

        # Get best detections (highest confidence)
        scale_boxes.sort(key=lambda x: x[1], reverse=True)
        scale_box, scale_conf = scale_boxes[0]

        underwater_scale_boxes.sort(key=lambda x: x[1], reverse=True)
        underwater_scale_box, underwater_scale_conf = underwater_scale_boxes[0]

        self.logger.info(f"Detected scale (conf={scale_conf:.3f}) and underwater_scale (conf={underwater_scale_conf:.3f})")

        # Initial waterline estimate: top edge of underwater_scale bbox
        waterline_y = underwater_scale_box[1]  # Top Y coordinate
        waterline_conf = underwater_scale_conf
        sam_refined = False

        # Refine waterline with SAM if enabled
        if self.segmentation_model is not None:
            self.logger.info("Attempting SAM refinement for precise waterline detection")
            refined_waterline_y = self._refine_waterline_with_sam(image, scale_box)
            if refined_waterline_y is not None:
                waterline_y = refined_waterline_y
                sam_refined = True
                self.logger.info(f"Using SAM-refined waterline at y={waterline_y:.1f}")
            else:
                self.logger.warning(f"SAM refinement failed, using underwater_scale top edge at y={waterline_y:.1f}")
        else:
            self.logger.info(f"SAM not enabled, using underwater_scale top edge at y={waterline_y:.1f}")

        # Calculate measurements
        measurements = self._calculate_water_level(
            image, waterline_y, underwater_scale_box, scale_box
        )

        # Create result dictionary
        result = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'water_level_cm': measurements['water_level_cm'],
            'scale_above_water_cm': measurements['scale_above_water_cm'],
            'confidence': float(waterline_conf),
            'detection_method': f"model_{self.config['models']['detection']['type']}",
            'sam_refined': sam_refined,
            'waterline_y': float(waterline_y),
            'underwater_scale_box': underwater_scale_box.tolist(),
            'scale_box': scale_box.tolist(),
            'scale_confidence': float(scale_conf),
            'underwater_scale_confidence': float(underwater_scale_conf),
            'success': True
        }

        self.logger.info(f"Water level: {result['water_level_cm']:.2f} cm, "
                        f"confidence: {result['confidence']:.3f}")

        return result

    def _calculate_water_level(self, image: np.ndarray, waterline_y: float,
                              underwater_scale_box: np.ndarray,
                              scale_box: np.ndarray) -> Dict[str, float]:
        """
        Calculate water level in centimeters.

        Method:
        1. Use calibrated pixels_per_cm (from above-water calibration)
        2. Calculate scale_above_water (from scale top to waterline) in cm
        3. Water level = total_scale_height - scale_above_water

        Args:
            image: Input image
            waterline_y: Y-coordinate of waterline (from SAM or underwater_scale top)
            underwater_scale_box: Underwater scale bounding box [x1, y1, x2, y2]
            scale_box: Total scale bounding box [x1, y1, x2, y2]

        Returns:
            Dictionary with measurements
        """
        # Use calibrated pixels_per_cm (REQUIRED - must be set via calibration)
        # DO NOT calculate from scale height as underwater portion is distorted by refraction
        if self.pixels_per_cm is None:
            self.logger.error("pixels_per_cm not calibrated! Run: python -m src.calibration")
            raise ValueError(
                "Calibration required! Run calibration:\n"
                "  python -m src.calibration\n"
                "Use an image with visible above-water scale markings."
            )

        # Get current image dimensions
        current_height, current_width = image.shape[:2]

        # Get calibration image dimensions
        calib_height = self.config.get('calibration', {}).get('image_height')
        calib_width = self.config.get('calibration', {}).get('image_width')

        # Scale the pixels_per_cm ratio if image dimensions differ from calibration
        if calib_height and calib_width:
            # Use height for scaling (more stable for vertical scales)
            scale_factor = current_height / calib_height
            pixels_per_cm = self.pixels_per_cm * scale_factor

            self.logger.info(f"Image dimensions: {current_width}x{current_height} (WxH)")
            self.logger.info(f"Calibration dimensions: {calib_width}x{calib_height}")
            self.logger.info(f"Scale factor: {scale_factor:.4f}")
            self.logger.info(f"Calibrated pixels_per_cm: {self.pixels_per_cm:.2f}")
            self.logger.info(f"Scaled pixels_per_cm for current image: {pixels_per_cm:.2f}")
        else:
            # No calibration dimensions available, use raw value
            pixels_per_cm = self.pixels_per_cm
            self.logger.warning("Calibration dimensions not found, using raw pixels_per_cm")
            self.logger.info(f"Image dimensions: {current_width}x{current_height} (WxH)")
            self.logger.info(f"Using calibrated pixels_per_cm: {pixels_per_cm:.2f}")

        self.logger.info(f"Scale bbox: top={scale_box[1]:.1f}, bottom={scale_box[3]:.1f}")
        self.logger.info(f"Waterline Y: {waterline_y:.1f}")

        # Calculate scale above water (from scale top to waterline)
        scale_top_y = scale_box[1]
        scale_above_water_pixels = waterline_y - scale_top_y
        scale_above_water_cm = scale_above_water_pixels / pixels_per_cm

        # Calculate water level (total scale height minus above-water portion)
        water_level_cm = self.scale_height_cm - scale_above_water_cm

        self.logger.info(f"Scale above water: {scale_above_water_pixels:.1f} pixels = {scale_above_water_cm:.2f} cm")
        self.logger.info(f"Water level: {water_level_cm:.2f} cm")

        return {
            'water_level_cm': water_level_cm,
            'scale_above_water_cm': scale_above_water_cm
        }

    def _create_failed_result(self, image_path: str, image: np.ndarray,
                             reason: str) -> Dict[str, Any]:
        """Create a failed detection result."""
        return {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'water_level_cm': 0.0,
            'scale_above_water_cm': 0.0,
            'confidence': 0.0,
            'detection_method': f"model_{self.config['models']['detection']['type']}",
            'success': False,
            'failure_reason': reason
        }

    def _save_annotated_image(self, image: np.ndarray, detections: Dict[str, Any],
                             result: Dict[str, Any], image_path: str):
        """Save annotated image with detections."""
        try:
            annotated = image.copy()
            boxes = detections['boxes']
            class_names = detections['class_names']
            confidences = detections['confidences']

            ann_config = self.config['output']['annotation']

            # Draw bounding boxes
            for i, (box, class_name, conf) in enumerate(zip(boxes, class_names, confidences)):
                x1, y1, x2, y2 = box.astype(int)

                # Color based on class
                if class_name.lower() in ['waterline', 'water_line', 'water']:
                    color = (0, 255, 0)  # Green for waterline
                elif class_name.lower() in ['scale', 'ruler']:
                    color = (255, 0, 0)  # Blue for scale
                else:
                    color = (0, 0, 255)  # Red for others

                if ann_config['draw_boxes']:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color,
                                ann_config['box_thickness'])

                if ann_config['draw_labels'] and ann_config['draw_confidence']:
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, ann_config['font_scale'],
                              color, 2)

            # Add measurement text
            if result['success']:
                text = f"Water Level: {result['water_level_cm']:.2f} cm"
                cv2.putText(annotated, text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "success" if result['success'] else "failed"
            filename = f"annotated_{status}_{timestamp}_{Path(image_path).stem}.jpg"
            output_path = self.annotated_dir / filename

            cv2.imwrite(str(output_path), annotated)
            self.logger.debug(f"Saved annotated image: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save annotated image: {e}")
