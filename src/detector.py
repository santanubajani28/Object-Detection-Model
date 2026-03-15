"""
detector.py — Core YOLO inference engine.

Loads multiple YOLO models (general + specialized fine-tuned),
runs inference per frame, and returns a unified list of Detection objects.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO

from src.utils import Detection, get_logger, load_config

logger = get_logger("Detector")

# ── COCO classes that map to our categories ────────────────────────────────────
SIGN_COCO_IDS   = {11}          # "stop sign" in COCO
VEHICLE_COCO_IDS= {2, 3, 5, 7}  # car, motorcycle, bus, truck
PERSON_COCO_ID  = {0}


class DetectionEngine:
    """
    Multi-model YOLO detection engine.

    It runs up to 4 models per frame:
      1. General (COCO) — vehicles, person, signs  [always on]
      2. Pothole model  [if weight file exists]
      3. Helmet model   [if weight file exists]
      4. License plate  [if weight file exists]
    """

    def __init__(self, cfg: dict):
        self._cfg      = cfg
        self._device   = cfg.get("device", "cpu")
        self._imgsz    = cfg.get("imgsz", 640)
        self._conf     = cfg.get("confidence", {})
        self._models_dir = Path("models")
        self._track_cfg = cfg.get("tracking", {})

        self.models: dict[str, Optional[YOLO]] = {
            "general":       None,
            "pothole":       None,
            "helmet":        None,
            "license_plate": None,
        }
        self._load_models(cfg.get("models", {}))

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_models(self, model_cfg: dict):
        for key, filename in model_cfg.items():
            self.models[key] = self._load_single(key, filename)

    def _load_single(self, key: str, filename: str) -> Optional[YOLO]:
        """
        Load one YOLO model weight.

        Resolution order:
          1. models/<filename>  — local file (if valid)
          2. Ultralytics auto-download — for 'general' key only (yolov8n.pt)
          3. Skip with warning      — for specialized models not yet available
        """
        path = self._models_dir / filename

        # ── Try local file first ───────────────────────────────────────────────
        if path.exists():
            try:
                logger.info(f"Loading model [{key}]: {path}")
                model = YOLO(str(path))
                # Warm-up
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                model.predict(dummy, imgsz=self._imgsz, device=self._device,
                               verbose=False, conf=0.01)
                logger.info(f"  ✓ [{key}] ready from local file")
                return model
            except (RuntimeError, Exception) as e:
                logger.warning(f"  ✗ [{key}] local file corrupted ({e})")
                logger.warning(f"    Deleting bad file: {path}")
                try:
                    path.unlink()
                except Exception:
                    pass
                # Fall through to auto-download logic below

        # ── Auto-download via ultralytics (general/base model only) ───────────
        # Ultralytics will download yolov8n.pt (or whatever name) from its CDN
        # and cache it in the ultralytics home dir. We then copy it to models/.
        base_names = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                      "yolov8l.pt", "yolov8x.pt",
                      "yolov8n-seg.pt", "yolo11n.pt"}

        if key == "general" or filename in base_names:
            try:
                logger.info(f"  ↓ [{key}] downloading '{filename}' via ultralytics…")
                model = YOLO(filename)   # ultralytics handles the download
                # Warm-up
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                model.predict(dummy, imgsz=self._imgsz, device=self._device,
                               verbose=False, conf=0.01)
                # Copy to our models/ directory for next run
                import shutil, torch
                ult_path = Path(torch.hub.get_dir()) / "ultralytics" / "assets" / filename
                if not ult_path.exists():
                    # Try ultralytics settings cache dir
                    from ultralytics.utils import SETTINGS
                    ult_path = Path(SETTINGS.get("weights_dir", ".")) / filename
                if ult_path.exists():
                    shutil.copy(ult_path, path)
                    logger.info(f"  ✓ [{key}] saved to {path}")
                else:
                    logger.info(f"  ✓ [{key}] running from ultralytics cache")
                return model
            except Exception as e:
                logger.error(f"  ✗ [{key}] auto-download failed: {e}")
                return None

        # ── Specialized model not available ────────────────────────────────────
        logger.warning(
            f"  ✗ [{key}] weight '{filename}' not found in models/.\n"
            f"    Download from Roboflow and place as models/{filename}\n"
            f"    Skipping {key} detection."
        )
        return None

    # ── Inference ──────────────────────────────────────────────────────────────
    def run(self, frame: np.ndarray) -> List[Detection]:
        """Run all loaded models on one frame. Return unified Detection list."""
        detections: List[Detection] = []

        if frame is None:
            return detections

        # 1. General model (COCO)
        if self.models["general"]:
            detections += self._run_general(frame)

        # 2. Pothole model
        if self.models["pothole"]:
            detections += self._run_specialized(
                frame, "pothole", conf=self._conf.get("pothole", 0.40)
            )

        # 3. Helmet model
        if self.models["helmet"]:
            detections += self._run_specialized(
                frame, "helmet", conf=self._conf.get("helmet", 0.50)
            )

        # 4. License plate model
        if self.models["license_plate"]:
            detections += self._run_specialized(
                frame, "license_plate", conf=self._conf.get("license_plate", 0.55)
            )

        return detections

    def _run_general(self, frame: np.ndarray) -> List[Detection]:
        model = self.models["general"]
        conf  = self._conf.get("general", 0.45)

        tracking_enabled = self._track_cfg.get("enabled", True)
        try:
            if tracking_enabled:
                results = model.track(
                    frame, imgsz=self._imgsz, device=self._device,
                    conf=conf, persist=True, verbose=False,
                    tracker="bytetrack.yaml",
                )
            else:
                results = model.predict(
                    frame, imgsz=self._imgsz, device=self._device,
                    conf=conf, verbose=False,
                )
        except Exception as e:
            logger.warning(f"General model inference error: {e}")
            return []

        return self._parse_results(results, model_key="general")

    def _run_specialized(self, frame: np.ndarray, model_key: str, conf: float) -> List[Detection]:
        model = self.models[model_key]
        try:
            results = model.predict(
                frame, imgsz=self._imgsz, device=self._device,
                conf=conf, verbose=False,
            )
        except Exception as e:
            logger.warning(f"{model_key} model inference error: {e}")
            return []
        return self._parse_results(results, model_key=model_key)

    def _parse_results(self, results, model_key: str) -> List[Detection]:
        detections = []
        if not results:
            return detections

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            names = result.names  # id -> class name dict

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else None

                raw_name = names.get(cls_id, f"class_{cls_id}")
                class_name = self._map_class(raw_name, model_key, cls_id)

                if class_name is None:
                    continue   # Filter out irrelevant COCO classes

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                ))

        return detections

    def _map_class(self, raw_name: str, model_key: str, cls_id: int) -> Optional[str]:
        """Map raw COCO class names to our internal class labels."""
        if model_key != "general":
            return raw_name.lower().replace(" ", "_")

        # For general COCO model, filter to relevant classes
        if cls_id in SIGN_COCO_IDS:
            return "sign_board"
        if cls_id in VEHICLE_COCO_IDS:
            return "vehicle"
        if cls_id in PERSON_COCO_ID:
            return "person"
        return None   # Skip all other COCO classes
