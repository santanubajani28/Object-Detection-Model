# 🚗 ODM — Road Object Detection System

Real-time road hazard and traffic violation detection from live video using Python, YOLOv8, and OpenCV, with a live web dashboard.

## 🎯 What It Detects

| Detection | Model |
|---|---|
| 🕳️ Potholes | Fine-tuned YOLOv8 |
| 🪧 Sign Boards | YOLOv8 COCO pretrained |
| 🔢 License Plates + OCR | YOLOv8 + EasyOCR |
| 🏍️ Bike Rider w/o Helmet | Fine-tuned YOLOv8 |
| 🚗 Wrong-Lane Vehicles | Lane detection + ByteTrack |

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
cd D:\ODM
pip install -r requirements.txt
```

### 2. Download model weights
```bash
python scripts/download_models.py
```

### 3. Run on a test video file
```bash
python main.py --source "data/sample_videos/your_video.mp4"
```

### 4. Open the web dashboard
Navigate to: **http://localhost:5000**

### 5. Run with webcam
```bash
python main.py --source 0
```

### 6. Run with IP camera (RTSP)
```bash
python main.py --source "rtsp://user:pass@192.168.1.100/stream"
```

### 7. Headless mode (web dashboard only, no OpenCV window)
```bash
python main.py --source 0 --no-display
```

---

## 📁 Project Structure

```
D:\ODM\
├── main.py                     ← Entry point
├── config.yaml                 ← All settings (thresholds, source, etc.)
├── requirements.txt
│
├── src/
│   ├── capture.py              ← Video capture (webcam/RTSP/file)
│   ├── detector.py             ← YOLOv8 multi-model inference engine
│   ├── lane_analyzer.py        ← Lane detection + wrong-way logic
│   ├── ocr_reader.py           ← License plate OCR (EasyOCR)
│   ├── annotator.py            ← Frame annotation & overlays
│   ├── alert.py                ← Rate-limited alerts (console + log + audio)
│   └── utils.py                ← Shared helpers, Detection dataclass
│
├── web/
│   ├── app.py                  ← Flask + Socket.IO dashboard server
│   ├── templates/index.html    ← Dashboard UI
│   └── static/
│       ├── css/dashboard.css
│       └── js/dashboard.js
│
├── models/                     ← YOLO .pt weight files (auto-downloaded)
├── scripts/
│   └── download_models.py      ← Model weight downloader
├── data/sample_videos/         ← Drop test videos here
└── logs/detections.log         ← Structured JSON alert log
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
video_source: 0              # 0=webcam, "rtsp://...", or "path/to/file.mp4"
device: "cpu"               # Change to "cuda" when GPU is available
confidence:
  general: 0.45
  pothole: 0.40
  helmet:  0.50
  license_plate: 0.55
```

---

## 🧠 Upgrading to Fine-Tuned Models

The fine-tuned weights for Pothole, Helmet, and License Plate detection are sourced from **Roboflow Universe** (free datasets). To get the best accuracy:

1. Visit the dataset link and export in **YOLOv8 format**
2. Run: `python scripts/train_custom.py --data dataset.yaml`  *(coming in next phase)*
3. Place the resulting `best.pt` in `models/` and update `config.yaml`

| Target | Free Dataset |
|---|---|
| Pothole | [Roboflow: pothole-detection](https://universe.roboflow.com/pothole-rfkqs/pothole-detection-kocqk) |
| Helmet  | [Roboflow: helmet-detection](https://universe.roboflow.com/new-workspace-s9s5s/helmet-detection-j9yua) |
| Plate   | [Roboflow: license-plate-recognition](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) |

---

## 🔑 Keyboard Shortcuts (OpenCV window)

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit |

---

## 📊 Dashboard Features

- **Live MJPEG stream** with bounding box annotations
- **Per-class counters** (Potholes, Signs, Helmets, Wrong-way, etc.)
- **Real-time detection list** via Socket.IO
- **Alert log** with severity (DANGER / WARNING / INFO)
- **Wrong-way vehicle flash** — entire screen flashes red

---

## 🐛 Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| Slow inference on CPU | Reduce `imgsz: 416` in config.yaml |
| EasyOCR first-run slow | First run downloads language models (~300MB) |
| No webcam | Check `video_source: 0` and camera drivers |
| RTSP stream drops | Auto-reconnect is built in; check network/camera |
