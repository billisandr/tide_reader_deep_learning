#!/usr/bin/env python3
"""
Main application entry point for water level measurement system.
Model-based detection using YOLO/RT-DETR with optional SAM post-processing.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import yaml
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv

from .detector import WaterLevelDetector
from .model_loader import ModelLoader, load_secrets
from .database import DatabaseManager
from .utils import setup_logging, get_unprocessed_images


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)


def get_directory_with_gui(title="Select directory"):
    """Show directory selection dialog."""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory


def get_model_file_with_gui(initial_dir="models/detection"):
    """Show file selection dialog for model."""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select detection model file",
        initialdir=initial_dir,
        filetypes=[
            ("PyTorch models", "*.pt"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return filepath


def prompt_user_choices(config):
    """
    Prompt user for runtime configuration choices.

    Returns:
        dict: User choices for input_dir, model_path, use_sam
    """
    choices = {}

    print("\n" + "="*60)
    print("Water Level Detection System - Configuration")
    print("="*60)

    # 1. Input directory selection
    print("\n[1] Input Directory Selection")
    print(f"    Default: {config['processing'].get('default_input_dir', 'data/input')}")

    if os.environ.get('USE_GUI_SELECTOR', 'false').lower() == 'true':
        use_gui = True
    else:
        response = input("    Use GUI to select directory? (y/n, default: n): ").strip().lower()
        use_gui = response == 'y'

    if use_gui:
        print("    Opening directory selector...")
        input_dir = get_directory_with_gui("Select input directory for images")
        if not input_dir:
            print("    No directory selected. Using default.")
            input_dir = config['processing'].get('default_input_dir', 'data/input')
    else:
        custom_path = input("    Enter custom path (or press Enter for default): ").strip()
        input_dir = custom_path if custom_path else config['processing'].get('default_input_dir', 'data/input')

    choices['input_dir'] = Path(input_dir)
    print(f"    Selected: {choices['input_dir']}")

    # 2. Model selection - only for local models
    model_source = config['models']['detection']['source']

    if model_source == 'local':
        print("\n[2] Detection Model Selection")
        default_model = config['models']['detection']['local_path']
        print(f"    Default: {default_model}")

        response = input("    Use GUI to select model? (y/n, default: n): ").strip().lower()

        if response == 'y':
            print("    Opening file selector...")
            model_path = get_model_file_with_gui()
            if not model_path:
                print("    No model selected. Using default.")
                model_path = default_model
        else:
            custom_model = input("    Enter custom model path (or press Enter for default): ").strip()
            model_path = custom_model if custom_model else default_model

        choices['model_path'] = model_path
        print(f"    Selected: {choices['model_path']}")
    else:
        # Using Roboflow - no local model needed
        print("\n[2] Detection Model")
        print(f"    Source: {model_source}")
        model_id = config['models']['detection'].get('roboflow_model_id', 'Not configured')
        print(f"    Model ID: {model_id}")
        print(f"    Using hosted inference (no local model required)")
        choices['model_path'] = None  # Not needed for Roboflow

    # 3. SAM post-processing
    print("\n[3] SAM Post-Processing")
    print("    SAM (Segment Anything Model) can refine detections for better accuracy")
    print("    Useful for: precise scale boundaries, underwater scale masking")

    sam_enabled = config['models']['segmentation'].get('enabled', False)
    default_choice = 'y' if sam_enabled else 'n'
    print(f"    Config setting: {'Enabled' if sam_enabled else 'Disabled'}")

    response = input(f"    Enable SAM post-processing? (y/n, default: {default_choice}): ").strip().lower()

    # Use config default if user just presses Enter, otherwise respect user choice
    if response == '':
        choices['use_sam'] = sam_enabled
    else:
        choices['use_sam'] = response == 'y'

    print(f"    SAM: {'Enabled' if choices['use_sam'] else 'Disabled'}")

    print("\n" + "="*60)
    print("Configuration complete. Starting detection system...")
    print("="*60 + "\n")

    return choices


def main():
    """Main application loop."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config()

    # Setup logging
    log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Prompt user for configuration choices
    user_choices = prompt_user_choices(config)

    # Override config with user choices
    if user_choices['model_path']:  # Only set if using local model
        config['models']['detection']['local_path'] = user_choices['model_path']
    config['models']['segmentation']['enabled'] = user_choices['use_sam']

    logger.info("="*60)
    logger.info("Water Level Detection System - Model-Based")
    logger.info("="*60)
    logger.info(f"Input directory: {user_choices['input_dir']}")

    # Log appropriate model info based on source
    model_source = config['models']['detection']['source']
    if model_source == 'local':
        logger.info(f"Model: {user_choices['model_path']}")
    else:
        model_id = config['models']['detection'].get('roboflow_model_id', 'Not configured')
        logger.info(f"Model source: {model_source}")
        logger.info(f"Model ID: {model_id}")

    logger.info(f"SAM enabled: {user_choices['use_sam']}")
    logger.info("="*60)

    # Load secrets (for Roboflow API)
    secrets = load_secrets()

    # Initialize model loader
    model_loader = ModelLoader(config, secrets)

    # Load detection model
    try:
        logger.info("Loading detection model...")
        detection_model = model_loader.load_detection_model()
        logger.info("Detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load detection model: {e}")
        logger.error("Please check your model path and ensure the file exists")
        sys.exit(1)

    # Load segmentation model (optional)
    segmentation_model = None
    if user_choices['use_sam']:
        try:
            logger.info("Loading SAM segmentation model...")
            segmentation_model = model_loader.load_segmentation_model()
            if segmentation_model:
                logger.info("SAM model loaded successfully")
            else:
                logger.warning("SAM model not available, continuing without it")
        except Exception as e:
            logger.warning(f"Failed to load SAM model: {e}")
            logger.warning("Continuing without segmentation")

    # Initialize detector
    detector = WaterLevelDetector(config, detection_model, segmentation_model)

    # Initialize database
    db_path = Path(config['database']['path'])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_manager = DatabaseManager(str(db_path))
    logger.info(f"Database initialized: {db_path}")

    # Setup directories
    input_dir = user_choices['input_dir']
    input_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path('data/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check export settings
    export_enabled = (
        config['output']['csv_export'] or
        config['output']['json_export'] or
        config['output']['database']
    )

    logger.info(f"Export formats - CSV: {config['output']['csv_export']}, "
               f"JSON: {config['output']['json_export']}, "
               f"DB: {config['output']['database']}")

    # Processing loop
    process_interval = int(os.environ.get('PROCESS_INTERVAL', 60))
    logger.info(f"Starting processing loop (interval: {process_interval}s)")
    logger.info(f"Monitoring directory: {input_dir}")
    logger.info("Press Ctrl+C to stop")
    logger.info("-"*60)

    while True:
        try:
            # Get unprocessed images
            images = get_unprocessed_images(input_dir, processed_dir, db_manager)

            if images:
                logger.info(f"Found {len(images)} new images to process")

                for image_path in images:
                    try:
                        # Process image
                        result = detector.process_image(str(image_path))

                        if result and result.get('success', False):
                            # Store result in database
                            db_manager.store_measurement(
                                timestamp=result['timestamp'],
                                water_level=result['water_level_cm'],
                                scale_above_water=result['scale_above_water_cm'],
                                image_path=str(image_path),
                                confidence=result['confidence'],
                                detection_method=result['detection_method']
                            )

                            logger.info(f"Processed {image_path.name}: "
                                      f"Water level = {result['water_level_cm']:.2f} cm, "
                                      f"Scale above water = {result['scale_above_water_cm']:.2f} cm")

                        else:
                            reason = result.get('failure_reason', 'Unknown') if result else 'Processing failed'
                            logger.warning(f"Failed to process {image_path.name}: {reason}")

                        # Move to processed directory
                        processed_path = processed_dir / image_path.name

                        # Copy in debug mode, move in normal mode
                        if os.environ.get('DEBUG_MODE', 'false').lower() == 'true':
                            import shutil
                            shutil.copy2(str(image_path), str(processed_path))
                            logger.debug(f"DEBUG: Copied {image_path.name} (original preserved)")
                        else:
                            image_path.rename(processed_path)
                            logger.debug(f"Moved {image_path.name} to processed")

                    except Exception as e:
                        logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
                        continue

                # Export data in configured formats
                if export_enabled:
                    try:
                        db_manager.export_all_formats(output_dir, config)
                        logger.info("Exported measurements in configured formats")
                    except Exception as e:
                        logger.error(f"Error exporting data: {e}")
            else:
                logger.debug("No new images to process")

            # Wait before next iteration
            time.sleep(process_interval)

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(process_interval)

    logger.info("Application stopped")


if __name__ == "__main__":
    main()
