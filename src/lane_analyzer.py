"""
lane_analyzer.py — Lane detection and wrong-way vehicle analysis.

Pipeline:
  1. Apply trapezoid ROI mask on the lower portion of the frame
  2. Grayscale → Gaussian blur → Canny edges
  3. HoughLinesP to find lane lines
  4. Classify into left / right lanes
  5. Track vehicle centroids across frames; if a vehicle moves
     against the expected traffic direction → flag WRONG_WAY
"""
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

from src.utils import Detection, get_logger

logger = get_logger("LaneAnalyzer")


def _make_roi_mask(frame_shape: Tuple[int, int], roi_cfg: dict) -> np.ndarray:
    """Build a polygon mask from relative ROI config points."""
    h, w = frame_shape[:2]
    tl = roi_cfg.get("top_left",     [0.35, 0.55])
    tr = roi_cfg.get("top_right",    [0.65, 0.55])
    br = roi_cfg.get("bottom_right", [0.95, 0.95])
    bl = roi_cfg.get("bottom_left",  [0.05, 0.95])
    pts = np.array([
        [int(tl[0]*w), int(tl[1]*h)],
        [int(tr[0]*w), int(tr[1]*h)],
        [int(br[0]*w), int(br[1]*h)],
        [int(bl[0]*w), int(bl[1]*h)],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


class LaneAnalyzer:
    """Detects lane lines and flags vehicles travelling in the wrong direction."""

    def __init__(self, cfg: dict):
        lane_cfg         = cfg.get("lane", {})
        self._enabled    = lane_cfg.get("enabled", True)
        self._roi_cfg    = lane_cfg.get("roi", {})
        self._canny_low  = lane_cfg.get("canny_low", 50)
        self._canny_high = lane_cfg.get("canny_high", 150)
        self._hough_thr  = lane_cfg.get("hough_threshold", 30)
        self._ww_thresh  = lane_cfg.get("wrong_way_vector_threshold", 0.6)

        # Track vehicle centroids: track_id → list of (x, y) positions
        self._history: dict[int, list] = defaultdict(list)
        self._wrong_way_ids: set       = set()
        self._HISTORY_LEN = 15   # frames to track

        # Cached detections with wrong-way flag
        self._wrong_way_detections: List[Detection] = []

    def analyze(
        self,
        frame: np.ndarray,
        vehicle_detections: List[Detection],
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Analyze frame for lanes and wrong-way vehicles.

        Returns:
            annotated_frame: frame with lane lines drawn
            wrong_way_list : list of Detection objects flagged as wrong-way
        """
        if not self._enabled or frame is None:
            return frame, []

        h, w = frame.shape[:2]

        # ── 1. Build lane-line overlay ─────────────────────────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blur, self._canny_low, self._canny_high)

        roi_mask, roi_pts = _make_roi_mask(frame.shape, self._roi_cfg)
        masked_edges      = cv2.bitwise_and(edges, edges, mask=roi_mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1, theta=np.pi / 180,
            threshold=self._hough_thr,
            minLineLength=40, maxLineGap=100,
        )

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.3:
                    left_lines.append(line[0])
                elif slope > 0.3:
                    right_lines.append(line[0])

        lane_frame = frame.copy()
        for ln in left_lines:
            cv2.line(lane_frame, (ln[0], ln[1]), (ln[2], ln[3]), (0, 255, 0), 2)
        for ln in right_lines:
            cv2.line(lane_frame, (ln[0], ln[1]), (ln[2], ln[3]), (0, 255, 0), 2)

        # Draw ROI boundary (debug, semi-transparent)
        overlay = lane_frame.copy()
        cv2.polylines(overlay, [roi_pts], True, (255, 255, 0), 1)
        cv2.addWeighted(overlay, 0.3, lane_frame, 0.7, 0, lane_frame)

        # ── 2. Wrong-way vehicle tracking ─────────────────────────────────────
        wrong_way_detections: List[Detection] = []
        mid_x = w // 2

        for det in vehicle_detections:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            history = self._history[det.track_id]
            history.append((cx, cy))
            if len(history) > self._HISTORY_LEN:
                history.pop(0)

            # Need at least 8 frames to judge direction
            if len(history) < 8:
                continue

            # Compute overall motion vector (dx, dy)
            dx = history[-1][0] - history[0][0]
            dy = history[-1][1] - history[0][1]

            # In a typical dashcam view traffic moves away (dy < 0 = up screen)
            # Wrong-way vehicles move toward camera (dy > 0 = down screen)
            # and appear on the right lane (cx > mid_x)
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude < 20:
                continue

            norm_dy = dy / magnitude
            # Heuristic: vehicle on right half of frame moving toward camera
            is_wrong = (norm_dy > self._ww_thresh) and (cx > mid_x)

            if is_wrong:
                self._wrong_way_ids.add(det.track_id)
                ww = Detection(
                    class_name="wrong_way",
                    confidence=min(0.95, abs(norm_dy)),
                    bbox=det.bbox,
                    track_id=det.track_id,
                    extra={"vehicle_class": det.class_name},
                )
                wrong_way_detections.append(ww)
                # Draw red border on frame
                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    lane_frame, "WRONG WAY!",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2,
                )
            else:
                self._wrong_way_ids.discard(det.track_id)

        return lane_frame, wrong_way_detections

    def cleanup_old_tracks(self, active_ids: set):
        """Remove history for tracks no longer active."""
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]
            self._wrong_way_ids.discard(tid)
