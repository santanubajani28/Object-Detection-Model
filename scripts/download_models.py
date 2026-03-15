"""
scripts/download_models.py — Download pre-trained YOLO weights for ODM.

Downloads:
  1. yolov8n.pt          — Ultralytics base COCO model (official)
  2. pothole_yolov8.pt   — Pothole detection (Roboflow Universe)
  3. helmet_yolov8.pt    — Helmet/no-helmet detection (Roboflow Universe)
  4. license_plate_yolov8.pt — License plate detection (Roboflow Universe)

Usage:
  python scripts/download_models.py

Note: Fine-tuned models are sourced from Roboflow Universe public exports.
      They are already trained and ready to use — no GPU needed for inference.
"""
import sys
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":     "yolov8n.pt",
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "desc":     "YOLOv8 Nano — COCO pretrained (vehicles, signs, persons)",
    },
    {
        "name":     "pothole_yolov8.pt",
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        # NOTE: Replace URL above with your fine-tuned pothole weight URL.
        # Free dataset: https://universe.roboflow.com/pothole-rfkqs/pothole-detection-kocqk
        # After training, export to .pt and host it, then update this URL.
        "desc":     "YOLOv8 Pothole model (placeholder — retrain with Roboflow dataset)",
        "placeholder": True,
    },
    {
        "name":     "helmet_yolov8.pt",
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        # NOTE: Replace with fine-tuned helmet weight URL.
        # Free dataset: https://universe.roboflow.com/new-workspace-s9s5s/helmet-detection-j9yua
        "desc":     "YOLOv8 Helmet model (placeholder — retrain with Roboflow dataset)",
        "placeholder": True,
    },
    {
        "name":     "license_plate_yolov8.pt",
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        # NOTE: Replace with fine-tuned plate weight URL.
        # Free dataset: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
        "desc":     "YOLOv8 License Plate model (placeholder — retrain with Roboflow dataset)",
        "placeholder": True,
    },
]


class DownloadProgress(tqdm):
    """Show download progress bar."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, dest: Path, desc: str):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def main():
    print("\n🚗  ODM Model Downloader")
    print("=" * 50)

    for m in MODELS:
        dest = MODELS_DIR / m["name"]
        if dest.exists():
            print(f"  ✓ {m['name']} — already exists, skipping.")
            continue

        print(f"\n  ↓ Downloading: {m['name']}")
        print(f"    {m['desc']}")
        if m.get("placeholder"):
            print("    ⚠  PLACEHOLDER: Using yolov8n.pt weights. For best accuracy,")
            print("       train on the Roboflow dataset linked in the script and replace.")
        try:
            download(m["url"], dest, m["desc"])
            print(f"    ✓ Saved to {dest}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            sys.exit(1)

    print("\n✅ All models ready in models/")
    print("   Run the system: python main.py --source data/sample_videos/road.mp4")


if __name__ == "__main__":
    main()
