"""
web/app.py — Flask web dashboard with MJPEG live stream and Socket.IO events.
"""
import cv2
import time
import threading
from pathlib import Path
from typing import Callable

from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO

_socketio_instance = None


def create_app(cfg: dict, frame_provider: Callable, dets_provider: Callable):
    """
    Factory that builds and returns (Flask app, SocketIO instance).

    frame_provider: callable() → numpy BGR frame or None
    dets_provider:  callable() → List[Detection]
    """
    global _socketio_instance

    template_dir = Path(__file__).parent / "templates"
    static_dir   = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config["SECRET_KEY"] = "odm-secret-2026"

    socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")
    _socketio_instance = socketio

    web_cfg     = cfg.get("web", {})
    quality     = web_cfg.get("stream_quality", 75)
    fps_limit   = web_cfg.get("stream_fps_limit", 25)
    frame_delay = 1.0 / fps_limit

    # ── MJPEG streaming generator ─────────────────────────────────────────────
    def generate_frames():
        while True:
            start = time.time()
            frame = frame_provider()
            if frame is not None:
                ret, buffer = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
            elapsed = time.time() - start
            remaining = frame_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ── Routes ─────────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/detections")
    def api_detections():
        dets = dets_provider()
        data = [
            {
                "class":      d.class_name,
                "confidence": round(d.confidence, 3),
                "track_id":   d.track_id,
                "bbox":       list(d.bbox),
                "extra":      d.extra,
            }
            for d in dets
        ]
        return jsonify(data)

    @app.route("/api/alerts")
    def api_alerts():
        """Return recent alerts from log file."""
        from src.alert import AlertManager
        # Re-read log file directly (stateless endpoint)
        log_path = Path(cfg.get("alerts", {}).get("log_file", "logs/detections.log"))
        try:
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            import json
            alerts = []
            for line in lines[-30:]:
                try:
                    alerts.append(json.loads(line))
                except Exception:
                    pass
            return jsonify(list(reversed(alerts)))
        except Exception:
            return jsonify([])

    # ── Background Socket.IO event emitter ────────────────────────────────────
    def emit_detections():
        import json
        while True:
            dets = dets_provider()
            if dets and socketio:
                payload = [
                    {
                        "class":      d.class_name,
                        "confidence": round(d.confidence, 3),
                        "track_id":   d.track_id,
                    }
                    for d in dets
                ]
                socketio.emit("detections", payload)
            socketio.sleep(0.5)   # Use socketio-aware sleep

    socketio.start_background_task(emit_detections)

    return app, socketio
