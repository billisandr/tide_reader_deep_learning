"""
Database management for storing water level measurements.
"""

import sqlite3
from datetime import datetime
import pandas as pd
import logging
import json
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                water_level_cm REAL,
                scale_above_water_cm REAL,
                image_path TEXT,
                confidence REAL,
                detection_method TEXT,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add detection_method column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE measurements ADD COLUMN detection_method TEXT")
            self.logger.info("Added detection_method column to existing database")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def store_measurement(self, timestamp, water_level, scale_above_water, 
                         image_path, confidence, detection_method=None):
        """Store a water level measurement."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO measurements 
            (timestamp, water_level_cm, scale_above_water_cm, image_path, confidence, detection_method)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, water_level, scale_above_water, image_path, confidence, detection_method))
        
        # Mark image as processed
        cursor.execute('''
            INSERT OR IGNORE INTO processed_images (image_path)
            VALUES (?)
        ''', (image_path,))
        
        conn.commit()
        conn.close()
        
        self.logger.debug(f"Stored measurement: {water_level:.1f}cm at {timestamp}")
    
    def is_image_processed(self, image_path):
        """Check if an image has been processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Debug: Show all processed images in database
        cursor.execute('SELECT image_path FROM processed_images')
        processed_paths = cursor.fetchall()
        self.logger.debug(f"Database contains {len(processed_paths)} processed images:")
        for path in processed_paths[:5]:  # Show first 5
            self.logger.debug(f"  - {path[0]}")
        
        cursor.execute('''
            SELECT COUNT(*) FROM processed_images
            WHERE image_path = ?
        ''', (str(image_path),))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        self.logger.debug(f"Checking path: {image_path} -> processed: {count > 0}")
        return count > 0
    
    def get_measurements(self, start_date=None, end_date=None):
        """Retrieve measurements within date range."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM measurements"
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def export_to_csv(self, output_path, start_date=None, end_date=None):
        """Export measurements to CSV."""
        df = self.get_measurements(start_date, end_date)
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported {len(df)} measurements to {output_path}")
    
    def export_to_json(self, output_path, start_date=None, end_date=None):
        """Export measurements to JSON."""
        df = self.get_measurements(start_date, end_date)
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert DataFrame to JSON with proper datetime handling
        measurements = []
        for _, row in df.iterrows():
            measurement = {
                'id': int(row['id']),
                'timestamp': row['timestamp'],
                'water_level_cm': float(row['water_level_cm']) if pd.notna(row['water_level_cm']) else None,
                'scale_above_water_cm': float(row['scale_above_water_cm']) if pd.notna(row['scale_above_water_cm']) else None,
                'image_path': row['image_path'],
                'confidence': float(row['confidence']) if pd.notna(row['confidence']) else None,
                'processed_at': row['processed_at']
            }
            measurements.append(measurement)
        
        with open(output_path, 'w') as f:
            json.dump(measurements, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(measurements)} measurements to {output_path}")
    
    def export_all_formats(self, output_dir, config, start_date=None, end_date=None):
        """Export measurements in all enabled formats based on config."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check config for enabled formats
        if config.get('output', {}).get('csv_export', False):
            csv_path = output_path / f"measurements_{timestamp}.csv"
            self.export_to_csv(csv_path, start_date, end_date)
        
        if config.get('output', {}).get('json_export', False):
            json_path = output_path / f"measurements_{timestamp}.json"
            self.export_to_json(json_path, start_date, end_date)
        
        # Database export (copy database file)
        if config.get('output', {}).get('database', False):
            db_backup_path = output_path / f"measurements_{timestamp}.db"
            import shutil
            shutil.copy2(self.db_path, db_backup_path)
            self.logger.info(f"Database backup created at {db_backup_path}")