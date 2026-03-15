"""
annotator.py — Frame annotation utilities.

Draws bounding boxes, labels, confidence scores, FPS counter,
and a live detection sidebar panel on each video frame.
"""
import cv2
import numpy as np
from typing import List

from src.utils import Detection, get_color, get_logger

logger = get_logger("Annotator")

# ── Label display names ────────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "pothole":       "🕳 Pothole",
    "sign_board":    "🪧 Sign Board",
    "license_plate": "🔢 Plate",
    "vehicle":       "🚗 Vehicle",
    "person":        "🚶 Person",
    "helmet":        "✅ Helmet",
    "no_helmet":     "❌ No Helmet",
    "wrong_way":     "⚠️ WRONG WAY",
}


class Annotator:
    """Draws all detection overlays on a frame."""

    def __init__(self, cfg: dict):
        self._show_fps   = cfg.get("show_fps", True)
        self._show_panel = cfg.get("show_detections_panel", True)

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Annotate frame with all detections.
        Returns a new annotated frame (does NOT modify in-place).
        """
        if frame is None:
            return frame

        out = frame.copy()

        for det in detections:
            self._draw_box(out, det)

        if self._show_fps:
            self._draw_fps(out, fps)

        if self._show_panel and detections:
            self._draw_panel(out, detections)

        return out

    # ── Private helpers ────────────────────────────────────────────────────────
    def _draw_box(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        color   = get_color(det.class_name)
        label   = DISPLAY_NAMES.get(det.class_name, det.class_name.replace("_", " ").title())
        conf_str = f"{det.confidence:.0%}"

        # Highlight wrong-way with thick red flashing border
        thickness = 4 if det.class_name == "wrong_way" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label background pill
        text = f"{label} {conf_str}"
        if det.track_id is not None:
            text += f"  #{det.track_id}"
        if "plate_text" in det.extra:
            text += f"  [{det.extra['plate_text']}]"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Pill fill
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, text,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    def _draw_fps(self, frame: np.ndarray, fps: float):
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, text,
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2,
            cv2.LINE_AA,
        )

    def _draw_panel(self, frame: np.ndarray, detections: List[Detection]):
        """Draw a translucent sidebar showing recent detections."""
        panel_w = 240
        panel_h = min(len(detections) * 28 + 40, frame.shape[0] - 20)
        x_off   = frame.shape[1] - panel_w - 10
        y_off   = 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_off), (x_off + panel_w, y_off + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "DETECTIONS", (x_off + 8, y_off + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        for i, det in enumerate(detections[:10]):
            y = y_off + 38 + i * 24
            color = get_color(det.class_name)
            label = DISPLAY_NAMES.get(det.class_name, det.class_name)
            # Colour dot
            cv2.circle(frame, (x_off + 12, y - 4), 5, color, -1)
            cv2.putText(frame, f"{label} {det.confidence:.0%}",
                        (x_off + 22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
