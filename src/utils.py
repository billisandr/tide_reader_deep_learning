"""
Utility functions for the water level measurement system.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import re

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'water_level.log'))
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    )

def get_unprocessed_images(input_dir, processed_dir, db_manager):
    """
    Get list of images that haven't been processed yet.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Getting unprocessed images from: {input_dir}")
    
    input_path = Path(input_dir)
    processed_path = Path(processed_dir)
    
    logger.debug(f"Input path exists: {input_path.exists()}")
    logger.debug(f"Processed path exists: {processed_path.exists()}")
    
    # Get all image files (use set to avoid duplicates on case-insensitive filesystems)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images_set = set()
    
    for ext in image_extensions:
        all_images_set.update(input_path.glob(ext))
    
    all_images = list(all_images_set)
    logger.debug(f"Found {len(all_images)} total image files")
    
    # Sort by filename (assuming timestamp in filename)
    all_images.sort(key=lambda x: extract_timestamp(x.name) or x.stat().st_mtime)
    
    # Filter out processed images
    unprocessed = []
    logger.debug("Checking database for processed images...")
    for img_path in all_images:
        logger.debug(f"Checking if processed: {img_path}")
        is_processed = db_manager.is_image_processed(str(img_path))
        logger.debug(f"Database says processed: {is_processed}")
        if not is_processed:
            unprocessed.append(img_path)
    
    logger.debug(f"Returning {len(unprocessed)} unprocessed images")
    return unprocessed

def extract_timestamp(filename):
    """
    Extract timestamp from filename if it follows a pattern.
    Example: IMG_20240101_120000.jpg
    """
    patterns = [
        r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
        r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        r'(\d{10,13})'  # Unix timestamp
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            timestamp_str = match.group(1)
            try:
                # Try parsing different formats
                if '_' in timestamp_str and len(timestamp_str) == 15:
                    return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                elif '-' in timestamp_str:
                    return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                elif len(timestamp_str) >= 10:
                    return datetime.fromtimestamp(int(timestamp_str[:10]))
            except:
                continue
    
    return None

def validate_image(image_path):
    """
    Validate that an image is suitable for processing.
    """
    import cv2
    
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Cannot read image"
        
        height, width = img.shape[:2]
        
        # Check minimum dimensions
        if width < 400 or height < 400:
            return False, f"Image too small: {width}x{height}"
        
        # Check if image is too dark or too bright
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        
        if mean_brightness < 30:
            return False, "Image too dark"
        elif mean_brightness > 225:
            return False, "Image too bright"
        
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

def create_summary_report(db_manager, output_dir, date_range=None):
    """
    Create a summary report of water level measurements.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime, timedelta

    # Get measurements
    if date_range:
        start_date, end_date = date_range
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
    
    df = db_manager.get_measurements(start_date, end_date)
    
    if df.empty:
        logging.warning("No measurements found for report")
        return None
    
    # Create report
    output_path = Path(output_dir)
    report_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Water level over time
    axes[0].plot(pd.to_datetime(df['timestamp']), df['water_level_cm'], 'b-')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Water Level (cm)')
    axes[0].set_title('Water Level Over Time')
    axes[0].grid(True)
    
    # Confidence scores
    axes[1].scatter(pd.to_datetime(df['timestamp']), df['confidence'], alpha=0.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Measurement Confidence')
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = output_path / f'water_level_report_{report_date}.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Generate statistics
    stats = {
        'period': f"{start_date.date()} to {end_date.date()}",
        'total_measurements': len(df),
        'average_water_level': df['water_level_cm'].mean(),
        'min_water_level': df['water_level_cm'].min(),
        'max_water_level': df['water_level_cm'].max(),
        'std_deviation': df['water_level_cm'].std(),
        'average_confidence': df['confidence'].mean()
    }
    
    # Save statistics to file
    stats_path = output_path / f'water_level_stats_{report_date}.txt'
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logging.info(f"Report generated: {plot_path} and {stats_path}")
    return stats