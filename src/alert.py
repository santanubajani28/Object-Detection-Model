"""
alert.py — Alert manager for road object detection events.

Features:
  - Rate-limited per-class alerts (cooldown window)
  - Structured JSON logging to file
  - Colored console output
  - Optional pygame audio alert
"""
import json
import time
import logging
from pathlib import Path
from typing import List
from collections import defaultdict

from src.utils import Detection, get_logger, ALERT_COLORS

logger = get_logger("Alert")


# ── Severity map ───────────────────────────────────────────────────────────────
SEVERITY = {
    "pothole":       "WARNING",
    "no_helmet":     "DANGER",
    "wrong_way":     "DANGER",
    "sign_board":    "INFO",
    "license_plate": "INFO",
    "vehicle":       "INFO",
    "person":        "INFO",
    "helmet":        "OK",
}


class AlertManager:
    """Rate-limited alert system with file logging and optional audio."""

    def __init__(self, cfg: dict):
        alert_cfg        = cfg.get("alerts", {})
        self._log_file   = Path(alert_cfg.get("log_file", "logs/detections.log"))
        self._cooldown   = alert_cfg.get("cooldown_seconds", 3)
        self._sound_on   = alert_cfg.get("sound_enabled", False)
        self._sound_file = alert_cfg.get("sound_file", "")

        # last trigger timestamp per class
        self._last_alert: dict[str, float] = defaultdict(float)

        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._setup_file_logger()

        # Init pygame audio (lazy)
        self._pygame_ready = False
        if self._sound_on:
            self._init_audio()

    def _setup_file_logger(self):
        self._file_logger = logging.getLogger("ODM.FileLog")
        if not self._file_logger.handlers:
            fh = logging.FileHandler(self._log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._file_logger.addHandler(fh)
            self._file_logger.propagate = False
        self._file_logger.setLevel(logging.INFO)

    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init()
            self._pygame_ready = True
        except Exception as e:
            logger.warning(f"Audio init failed: {e}")

    def process(self, detections: List[Detection]):
        """Process a list of detections and fire alerts as needed."""
        now = time.time()
        for det in detections:
            cls = det.class_name
            last = self._last_alert[cls]
            if now - last < self._cooldown:
                continue

            self._last_alert[cls] = now
            severity = SEVERITY.get(cls, "INFO")
            self._fire(det, severity)

    def _fire(self, det: Detection, severity: str):
        """Emit one alert (console + log file + optional sound)."""
        color = ALERT_COLORS.get(severity, "")
        msg   = (
            f"[{severity}] {det.class_name.upper().replace('_',' ')}  "
            f"conf={det.confidence:.0%}"
        )
        if det.track_id is not None:
            msg += f"  track#{det.track_id}"
        if "plate_text" in det.extra:
            msg += f"  plate={det.extra['plate_text']}"

        # Console
        print(f"{color}{msg}")

        # Structured JSON to log file
        entry = {
            "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "severity":   severity,
            "class":      det.class_name,
            "confidence": round(det.confidence, 3),
            "track_id":   det.track_id,
            "extra":      det.extra,
        }
        self._file_logger.info(json.dumps(entry))

        # Audio
        if self._sound_on and self._pygame_ready and severity in ("WARNING", "DANGER"):
            self._play_sound()

    def _play_sound(self):
        try:
            import pygame
            if self._sound_file and Path(self._sound_file).exists():
                pygame.mixer.music.load(self._sound_file)
                pygame.mixer.music.play()
            else:
                # Fallback: system beep via print
                print("\a", end="", flush=True)
        except Exception:
            pass

    def get_recent_alerts(self, limit: int = 20) -> list:
        """Read last N lines from log file for dashboard display."""
        try:
            lines = self._log_file.read_text(encoding="utf-8").strip().split("\n")
            parsed = []
            for line in lines[-limit:]:
                try:
                    parsed.append(json.loads(line))
                except Exception:
                    pass
            return list(reversed(parsed))
        except Exception:
            return []
