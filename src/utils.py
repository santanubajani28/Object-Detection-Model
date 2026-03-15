"""
utils.py — Shared helpers: FPS counter, colour palette, config loader, logger setup.
"""
import time
import logging
import yaml
from pathlib import Path
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ── Colour palette for bounding-box classes ────────────────────────────────────
CLASS_COLORS = {
    "pothole":         (0,   0,   255),   # Red (BGR)
    "sign":            (0,   200, 255),   # Yellow
    "sign board":      (0,   200, 255),
    "license_plate":   (255, 140, 0),     # Blue-ish
    "number_plate":    (255, 140, 0),
    "helmet":          (0,   230, 0),     # Green
    "no_helmet":       (0,   0,   255),   # Red
    "wrong_way":       (0,   0,   255),   # Red
    "vehicle":         (200, 200, 200),   # Grey
    "default":         (100, 255, 100),   # Lime
}

ALERT_COLORS = {
    "INFO":    Fore.CYAN,
    "WARNING": Fore.YELLOW,
    "DANGER":  Fore.RED + Style.BRIGHT,
    "OK":      Fore.GREEN,
}


def get_color(class_name: str):
    """Return BGR tuple for the given class label."""
    name = class_name.lower().replace(" ", "_")
    return CLASS_COLORS.get(name, CLASS_COLORS["default"])


# ── Config loader ──────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML config file as a dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ── Logger factory ─────────────────────────────────────────────────────────────
def get_logger(name: str = "ODM", level=logging.INFO) -> logging.Logger:
    """Create and return a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── FPS Counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self.window = window
        self._times: list[float] = []
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) > self.window:
            self._times.pop(0)
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0


# ── Detection dataclass ────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple           # (x1, y1, x2, y2) in pixel coords
    track_id: Optional[int] = None
    extra: dict = field(default_factory=dict)   # e.g., {"plate_text": "MH01AB1234"}
