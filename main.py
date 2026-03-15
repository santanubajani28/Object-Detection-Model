"""
main.py — Entry point for the Road Object Detection System.

Usage examples:
  # Run with webcam (index 0)
  python main.py

  # Run with a local video file
  python main.py --source data/sample_videos/road.mp4

  # Run as headless (no OpenCV window) — web dashboard only
  python main.py --no-display

  # Specify custom config
  python main.py --config config.yaml
"""
import cv2
import sys
import time
import argparse
import threading
from pathlib import Path

from src.utils      import load_config, FPSCounter, get_logger
from src.capture    import VideoCapture
from src.detector   import DetectionEngine
from src.lane_analyzer import LaneAnalyzer
from src.ocr_reader import OCRReader
from src.annotator  import Annotator
from src.alert      import AlertManager

logger = get_logger("Main")

# ── Shared state (accessed by Flask thread and main thread) ───────────────────
_frame_lock    = threading.Lock()
_latest_frame  = None         # latest annotated frame (numpy BGR)
_latest_dets   = []           # latest Detection list
_running       = True


def _update_shared(frame, dets):
    global _latest_frame, _latest_dets
    with _frame_lock:
        _latest_frame = frame.copy() if frame is not None else None
        _latest_dets  = list(dets)


def _start_web_dashboard(cfg):
    """Launch Flask web dashboard in a daemon thread."""
    try:
        from web.app import create_app
        app, socketio = create_app(cfg, frame_provider=_get_shared_frame,
                                   dets_provider=_get_shared_dets)
        web_cfg = cfg.get("web", {})
        host = web_cfg.get("host", "0.0.0.0")
        port = web_cfg.get("port", 5000)
        logger.info(f"Web dashboard starting at http://{host}:{port}")
        socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Web dashboard failed to start: {e}")


def _get_shared_frame():
    with _frame_lock:
        return _latest_frame


def _get_shared_dets():
    with _frame_lock:
        return list(_latest_dets)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(cfg: dict, source, display: bool = True):
    global _running

    fps_counter = FPSCounter(window=30)
    engine      = DetectionEngine(cfg)
    lane_ana    = LaneAnalyzer(cfg)
    ocr         = OCRReader(cfg)
    annotator   = Annotator(cfg)
    alert_mgr   = AlertManager(cfg)

    logger.info(f"Opening source: {source}")
    with VideoCapture(source, cfg.get("display_width", 1280), cfg.get("display_height", 720)) as cap:
        while _running:
            frame = cap.read()
            if frame is None:
                logger.warning("No frame received — retrying…")
                time.sleep(0.05)
                continue

            # ── 1. Object detection ────────────────────────────────────────────
            detections = engine.run(frame)

            # ── 2. OCR for license plates ──────────────────────────────────────
            for det in detections:
                if det.class_name == "license_plate":
                    x1, y1, x2, y2 = det.bbox
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    text = ocr.read_plate(crop, track_id=det.track_id or -1)
                    if text:
                        det.extra["plate_text"] = text

            # ── 3. Lane / wrong-way analysis ───────────────────────────────────
            vehicle_dets = [d for d in detections if d.class_name == "vehicle"]
            frame, wrong_way = lane_ana.analyze(frame, vehicle_dets)
            all_dets = detections + wrong_way

            # ── 4. Cleanup stale tracks ────────────────────────────────────────
            active_ids = {d.track_id for d in detections if d.track_id is not None}
            lane_ana.cleanup_old_tracks(active_ids)

            # ── 5. Annotate ────────────────────────────────────────────────────
            fps = fps_counter.tick()
            annotated = annotator.draw(frame, all_dets, fps)

            # ── 6. Alerts ─────────────────────────────────────────────────────
            alert_mgr.process(all_dets)

            # ── 7. Share with web dashboard ────────────────────────────────────
            _update_shared(annotated, all_dets)

            # ── 8. Local display window ────────────────────────────────────────
            if display:
                cv2.imshow("Road Object Detection — ODM", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Quit key pressed.")
                    _running = False
                    break

    cv2.destroyAllWindows()
    logger.info("Pipeline stopped.")


# ── CLI entry ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Road Object Detection System")
    parser.add_argument("--source",     default=None,
                        help="Video source: 0=webcam, RTSP URL, or file path")
    parser.add_argument("--config",     default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable local OpenCV window (headless / web-only mode)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    source = args.source if args.source is not None else cfg.get("video_source", 0)

    # Convert "0" string to int for webcam
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # ── Sanitize path: strip accidental relative prefix from absolute path ─────
    # e.g.  "data/sample_videos/C:/Users/.../video.mp4"  →  "C:/Users/.../video.mp4"
    if isinstance(source, str) and not source.startswith("rtsp"):
        import re as _re
        source = _re.sub(r'^[^/\\]*[/\\]+(?=[A-Za-z]:/)', '', source)
        source = source.replace("\\", "/")
        p = Path(source)
        if not p.is_absolute() and source not in ("0","1","2"):
            # Relative path — resolve from cwd
            source = str(Path.cwd() / source)
        if not Path(source).exists() and not source.startswith("rtsp"):
            logger.error(f"Video file not found: {source}")
            logger.error("Tip: use the full absolute path, e.g.")
            logger.error('  python main.py --source "C:/Users/santa/dwhelper/sample1.mp4"')
            raise SystemExit(1)
        logger.info(f"Resolved video source: {source}")

    display = not args.no_display

    # ── Start web dashboard in background thread ───────────────────────────────
    web_thread = threading.Thread(target=_start_web_dashboard, args=(cfg,), daemon=True)
    web_thread.start()

    # Small delay to let Flask bind
    time.sleep(1.0)

    try:
        run(cfg, source, display)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
