"""
capture.py — Video source abstraction.

Supports:
  - Webcam   : source = 0  (or any integer camera index)
  - IP Camera: source = "rtsp://user:pass@192.168.1.100/stream"
  - File     : source = "path/to/video.mp4"
"""
import cv2
import time
from src.utils import get_logger

logger = get_logger("Capture")


class VideoCapture:
    """Unified video capture wrapper with auto-reconnect for RTSP streams."""

    def __init__(self, source, width: int = 1280, height: int = 720, retry_delay: float = 2.0):
        self.source = source
        self.width  = width
        self.height = height
        self.retry_delay = retry_delay
        self._cap = None
        self._is_file = isinstance(source, str) and not source.startswith("rtsp")
        self._open()

    def _open(self):
        logger.info(f"Opening video source: {self.source}")
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.startswith("rtsp")):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_ANY)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")

        # Try setting resolution (works for cameras, ignored for files)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(
            f"Capture opened — "
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
            f"{self._cap.get(cv2.CAP_PROP_FPS):.1f}fps"
        )

    def read(self):
        """Return the next frame as a numpy array (BGR), or None on failure."""
        if self._cap is None:
            return None

        ret, frame = self._cap.read()

        # For files: loop back to start when video ends
        if not ret and self._is_file:
            logger.info("Video ended — looping back to start.")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()

        # For RTSP streams: attempt reconnect
        if not ret and not self._is_file:
            logger.warning("Stream read failed — attempting reconnect...")
            self._cap.release()
            time.sleep(self.retry_delay)
            try:
                self._open()
                ret, frame = self._cap.read()
            except IOError:
                logger.error("Reconnect failed.")
                return None

        return frame if ret else None

    def get_fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 30.0

    def get_resolution(self) -> tuple:
        if not self._cap:
            return (self.width, self.height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self):
        if self._cap:
            self._cap.release()
            logger.info("Video capture released.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
