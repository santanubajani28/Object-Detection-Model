"""
ocr_reader.py — EasyOCR-based license plate text extraction.

Receives a cropped plate region (numpy BGR image), runs OCR,
validates the text with a loose regex, and caches results to
avoid flickering between frames.
"""
import re
import cv2
import numpy as np
from collections import deque
from typing import Optional

from src.utils import get_logger

logger = get_logger("OCR")

# Loose alphanumeric plate regex (covers most world formats)
_PLATE_RE = re.compile(r"[A-Z0-9][A-Z0-9\s\-\.]{3,12}[A-Z0-9]", re.IGNORECASE)


class OCRReader:
    """Lazy-loads EasyOCR and provides plate text extraction."""

    def __init__(self, cfg: dict):
        ocr_cfg          = cfg.get("ocr", {})
        self._enabled    = ocr_cfg.get("enabled", True)
        self._languages  = ocr_cfg.get("languages", ["en"])
        self._min_conf   = ocr_cfg.get("min_confidence", 0.4)
        self._cache_len  = ocr_cfg.get("cache_frames", 10)

        self._reader     = None          # lazy load
        self._cache: dict[int, deque]   = {}   # track_id → deque of texts

    def _get_reader(self):
        if self._reader is None:
            logger.info("Loading EasyOCR model (first use)…")
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=False)
            logger.info("EasyOCR ready")
        return self._reader

    def read_plate(self, crop: np.ndarray, track_id: int = -1) -> Optional[str]:
        """
        Extract licence-plate text from a cropped BGR image.
        Returns cleaned plate string, or None if nothing valid found.
        """
        if not self._enabled or crop is None or crop.size == 0:
            return None

        # Pre-process: resize, grayscale, threshold
        h, w = crop.shape[:2]
        scale = max(1, 100 // h)
        crop_resized = cv2.resize(crop, (w * scale, h * scale))
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            reader  = self._get_reader()
            results = reader.readtext(thresh, detail=1)
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
            return self._from_cache(track_id)

        texts = []
        for (_, text, conf) in results:
            if conf >= self._min_conf:
                cleaned = re.sub(r"[^A-Z0-9\-]", "", text.upper())
                if _PLATE_RE.match(cleaned):
                    texts.append(cleaned)

        if texts:
            plate = max(texts, key=len)
            self._update_cache(track_id, plate)
            return plate

        return self._from_cache(track_id)

    def _update_cache(self, track_id: int, text: str):
        if track_id not in self._cache:
            self._cache[track_id] = deque(maxlen=self._cache_len)
        self._cache[track_id].append(text)

    def _from_cache(self, track_id: int) -> Optional[str]:
        """Return the most common recent reading for this track."""
        cache = self._cache.get(track_id)
        if not cache:
            return None
        # Majority vote from recent readings
        from collections import Counter
        counts = Counter(cache)
        return counts.most_common(1)[0][0]

    def clear_cache(self, track_id: int):
        self._cache.pop(track_id, None)
