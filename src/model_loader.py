"""
Model loader module for loading detection and segmentation models.
Supports local models and Roboflow API integration.
"""

import os
import logging
from pathlib import Path
import yaml
import torch
from typing import Optional, Dict, Any


class ModelLoader:
    """Handles loading of detection and segmentation models."""

    def __init__(self, config: Dict[str, Any], secrets: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelLoader.

        Args:
            config: Configuration dictionary from config.yaml
            secrets: Secrets dictionary from secrets.yaml (optional)
        """
        self.config = config
        self.secrets = secrets
        self.logger = logging.getLogger(__name__)

        self.detection_model = None
        self.segmentation_model = None

    def load_detection_model(self):
        """Load the detection model based on configuration."""
        detection_config = self.config['models']['detection']
        model_type = detection_config['type']
        source = detection_config['source']

        self.logger.info(f"Loading detection model: type={model_type}, source={source}")

        if source == 'local':
            return self._load_local_detection_model(detection_config)
        elif source == 'roboflow':
            return self._load_roboflow_model(detection_config)
        elif source == 'roboflow-inference':
            return self._load_roboflow_inference_model(detection_config)
        elif source == 'roboflow-serverless':
            return self._load_roboflow_serverless_model(detection_config)
        else:
            raise ValueError(f"Unknown model source: {source}")

    def _load_local_detection_model(self, detection_config: Dict[str, Any]):
        """Load a local detection model."""
        model_path = detection_config['local_path']
        model_type = detection_config['type']

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please place your model in the specified path or update config.yaml"
            )

        self.logger.info(f"Loading local model from: {model_path}")

        try:
            if model_type == 'yolo':
                from ultralytics import YOLO
                model = YOLO(model_path)
                self.logger.info(f"YOLO model loaded successfully")
            elif model_type == 'rt-detr':
                # RT-DETR is also supported by ultralytics
                from ultralytics import RTDETR
                model = RTDETR(model_path)
                self.logger.info(f"RT-DETR model loaded successfully")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.detection_model = model
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_roboflow_model(self, detection_config: Dict[str, Any]):
        """Load a model from Roboflow."""
        if not self.secrets or 'roboflow' not in self.secrets:
            raise ValueError(
                "Roboflow API key not found in secrets.yaml\n"
                "Please copy secrets.yaml.template to secrets.yaml and add your API key"
            )

        try:
            from roboflow import Roboflow

            api_key = self.secrets['roboflow']['api_key']
            workspace = detection_config['roboflow_workspace']
            project = detection_config['roboflow_project']
            version = detection_config['roboflow_version']

            self.logger.info(f"Connecting to Roboflow: {workspace}/{project}/{version}")

            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project)
            model = project.version(version).model

            self.logger.info(f"Roboflow model loaded successfully")

            self.detection_model = model
            return model

        except Exception as e:
            self.logger.error(f"Failed to load Roboflow model: {e}")
            raise

    def _load_roboflow_inference_model(self, detection_config: Dict[str, Any]):
        """Load a model using Roboflow Inference API (hosted inference via roboflow package)."""
        if not self.secrets or 'roboflow' not in self.secrets:
            raise ValueError(
                "Roboflow API key not found in secrets.yaml\n"
                "Please copy secrets.yaml.template to secrets.yaml and add your API key"
            )

        try:
            from roboflow import Roboflow

            api_key = self.secrets['roboflow']['api_key']
            model_id = detection_config.get('roboflow_model_id', '')

            # Parse model_id
            if not model_id:
                # Construct from workspace/project/version if not provided
                workspace = detection_config.get('roboflow_workspace', '')
                project = detection_config.get('roboflow_project', '')
                version = detection_config.get('roboflow_version', 1)

                if workspace and project:
                    model_id = f"{workspace}/{project}/{version}"
                else:
                    raise ValueError(
                        "Either 'roboflow_model_id' or 'roboflow_workspace' + "
                        "'roboflow_project' must be specified in config.yaml"
                    )

            # Parse model_id into components
            parts = model_id.split('/')
            if len(parts) < 2:
                raise ValueError(f"Invalid model_id format: {model_id}. Expected 'workspace/project' or 'workspace/project/version'")

            workspace = parts[0]
            project = parts[1]
            version = int(parts[2]) if len(parts) > 2 else 1

            self.logger.info(f"Connecting to Roboflow Inference API: {workspace}/{project}/{version}")

            # Initialize Roboflow
            rf = Roboflow(api_key=api_key)

            # Access model
            project_obj = rf.workspace(workspace).project(project)
            model = project_obj.version(version).model

            self.logger.info(f"Roboflow Inference model loaded successfully")
            self.logger.info(f"Using hosted inference (no local model download required)")

            self.detection_model = model
            return model

        except Exception as e:
            self.logger.error(f"Failed to load Roboflow Inference model: {e}")
            raise

    def _load_roboflow_serverless_model(self, detection_config: Dict[str, Any]):
        """Load a model using Roboflow Serverless API (inference-sdk)."""
        if not self.secrets or 'roboflow' not in self.secrets:
            raise ValueError(
                "Roboflow API key not found in secrets.yaml\n"
                "Please copy secrets.yaml.template to secrets.yaml and add your API key"
            )

        try:
            from inference_sdk import InferenceHTTPClient

            api_key = self.secrets['roboflow']['api_key']
            api_url = detection_config.get('roboflow_serverless_url', 'https://serverless.roboflow.com')
            model_id = detection_config.get('roboflow_model_id', '')

            if not model_id:
                raise ValueError(
                    "roboflow_model_id must be specified in config.yaml for serverless API\n"
                    "Format: 'project/version' (e.g., 'tide-recorder/5')"
                )

            self.logger.info(f"Connecting to Roboflow Serverless API: {api_url}")
            self.logger.info(f"Model ID: {model_id}")

            # Create inference client
            client = InferenceHTTPClient(
                api_url=api_url,
                api_key=api_key
            )

            self.logger.info(f"Roboflow Serverless client initialized successfully")
            self.logger.info(f"Using serverless inference (no local model download required)")

            # Store both client and model_id as we need model_id for inference
            self.detection_model = {
                'client': client,
                'model_id': model_id
            }

            return self.detection_model

        except Exception as e:
            self.logger.error(f"Failed to load Roboflow Serverless model: {e}")
            raise

    def load_segmentation_model(self):
        """Load the segmentation model (SAM) if enabled."""
        seg_config = self.config['models']['segmentation']

        if not seg_config.get('enabled', False):
            self.logger.info("Segmentation model disabled in config")
            return None

        model_path = seg_config['local_path']
        checkpoint_type = seg_config['checkpoint_type']

        if not os.path.exists(model_path):
            self.logger.warning(
                f"SAM checkpoint not found: {model_path}\n"
                f"Segmentation will be disabled. Download from: "
                f"https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
            return None

        try:
            from segment_anything import sam_model_registry, SamPredictor

            self.logger.info(f"Loading SAM model from: {model_path}")

            sam = sam_model_registry[checkpoint_type](checkpoint=model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device=device)

            predictor = SamPredictor(sam)

            self.logger.info(f"SAM model loaded successfully on {device}")

            self.segmentation_model = predictor
            return predictor

        except Exception as e:
            self.logger.error(f"Failed to load SAM model: {e}")
            self.logger.warning("Continuing without segmentation model")
            return None

    def get_detection_model(self):
        """Get the loaded detection model."""
        if self.detection_model is None:
            self.load_detection_model()
        return self.detection_model

    def get_segmentation_model(self):
        """Get the loaded segmentation model."""
        if self.segmentation_model is None:
            self.load_segmentation_model()
        return self.segmentation_model


def load_secrets(secrets_path: str = 'secrets.yaml') -> Optional[Dict[str, Any]]:
    """
    Load secrets from secrets.yaml file.

    Args:
        secrets_path: Path to secrets file

    Returns:
        Secrets dictionary or None if file doesn't exist
    """
    if not os.path.exists(secrets_path):
        logging.getLogger(__name__).warning(
            f"Secrets file not found: {secrets_path}\n"
            f"Copy secrets.yaml.template to secrets.yaml to use Roboflow integration"
        )
        return None

    try:
        with open(secrets_path, 'r') as f:
            secrets = yaml.safe_load(f)
        return secrets
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load secrets: {e}")
        return None
