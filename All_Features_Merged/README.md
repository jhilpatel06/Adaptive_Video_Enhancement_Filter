# Adaptive Video Enhancement Filter

A unified, real-time video processing pipeline that combines **AI-powered super-resolution**, **adaptive low-light enhancement**, **face-tracking filters**, and **gesture-controlled effects** into a single modular application.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)

---

## Features

### 🎨 Image Filters (Face-Aware & Global)
| Filter | Type | Description |
|---|---|---|
| **Snapchat Prototype** | Face | Googly eyes, clown nose, joker smile with adaptive scaling |
| **Beautify** | Face | Skin smoothing + saturation boost |
| **Sepia** | Global | Classic warm-tone vintage effect |
| **Grayscale** | Global | Black & white conversion |
| **Cartoon** | Global | Edge-detected cartoon rendering |
| **Cyberpunk** | Global | Chromatic aberration + neon color grading |
| **Thermal** | Global | Heat-map camera effect |
| **Pencil Sketch** | Global | Hand-drawn pencil sketch style |
| **Gesture Temperature** | Gesture | Cool/warm color shift controlled by hand gestures |

### 🧠 AI Super-Resolution (FSRCNN)
- **Face-based filters**: Automatically upscales small/distant faces using FSRCNN before applying the filter, then blends them back seamlessly with elliptical alpha masking.
- **Global filters**: Upsamples the entire filtered output to produce high-resolution results.

### 🌙 Adaptive Intensity Optimization
- **Gamma correction** lifts dark regions while protecting highlights.
- **CLAHE** improves local contrast on the luminance channel.
- **Bilateral noise filtering** reduces webcam noise while preserving edges.
- **Highlight suppression** compresses clipped bright regions.
- **Unsharp masking** restores midtone edge detail after denoising.
- **Temporal tone smoothing** stabilizes correction across frames to reduce flicker.
- **Performance-aware mode switching** reduces processing when frames become expensive.

### 🖐️ Gesture Control
- Live hand tracking via MediaPipe.
- Pinch thumb + index finger to shift color temperature from cool (blue) to warm (red).
- Interactive thermometer-style visual scale on screen.

### 👥 Multi-Face Tracking
- Processes all detected faces simultaneously.
- Stable face IDs via left-to-right spatial sorting.
- Per-face temporal smoothing (EMA) to eliminate jitter.

---

## Project Structure

```
All_Features_Merged/
├── main.py                          # Entry point — webcam loop, filter switching, display
├── pipeline.py                      # Central orchestrator (detection → enhancement → filter → blend)
├── requirements.txt
├── .gitignore
│
├── core/                            # Detection & enhancement modules
│   ├── face_detector.py             # MediaPipe face detection wrapper
│   ├── enhancer.py                  # FSRCNN super-resolution wrapper
│   ├── gesture_detector.py          # MediaPipe hand tracking + temperature mapping
│   └── intensity/                   # Adaptive intensity optimization package
│       ├── analyzer.py              # Frame quality analysis (brightness, contrast, motion, etc.)
│       ├── config.py                # Enhancement configuration & thresholds
│       ├── enhancer.py              # Adaptive gamma, CLAHE, noise filter, highlight suppression
│       └── utils.py                 # Overlay drawing, heatmap, histogram composition
│
├── filters/                         # All visual filter effects
│   ├── registry.py                  # FilterRegistry — register/lookup filters by name
│   ├── image_filters.py             # Sepia, grayscale, cartoon, cyberpunk, snapchat, etc.
│   └── gesture_filters.py           # Temperature color shift + thermometer UI
│
└── utils/                           # Shared helpers (reserved for future use)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Adaptive-Video-Enhancement-Filter.git
cd Adaptive-Video-Enhancement-Filter/All_Features_Merged
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model files

The following model files must be placed in the project root directory (`All_Features_Merged/`):

| File | Source |
|---|---|
| `blaze_face_full_range.tflite` | [MediaPipe Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models) |
| `face_landmarker.task` | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models) |
| `hand_landmarker.task` | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models) |
| `FSRCNN_x4.pb` | [OpenCV Super Resolution Models](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models) |

> These files are excluded from the repository via `.gitignore` because they are large binaries.

### 5. Run

```bash
python3 main.py
```

---

## Controls

| Key | Action |
|---|---|
| `1` – `9` | Switch to a specific filter (numbered in registration order) |
| `n` | Next filter |
| `p` | Previous filter |
| `m` | Toggle metrics overlay (brightness, contrast, FPS, mode) |
| `h` | Toggle luminance heatmap comparison |
| `g` | Toggle intensity histogram graph |
| `q` | Quit |

---

## Pipeline Architecture

```
Raw Frame
    │
    ▼
┌──────────────────────────────┐
│  Phase 0: Intensity Optimizer │  ← Gamma, CLAHE, denoise, highlight suppression
└──────────────┬───────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
  Gesture    Face      Global
  Detect    Detect     Filter
    │          │          │
    ▼          ▼          ▼
  Temp      FSRCNN     Apply
  Filter    + Filter   Filter
    │       + Blend      │
    ▼          │        FSRCNN
  FSRCNN       │       Upsample
  Upsample     ▼          │
    │       Final         ▼
    ▼       Frame       Final
  Final                 Frame
  Frame
               │
               ▼
        ┌─────────────┐
        │  UI Overlays │  ← Metrics, heatmap, histogram
        └──────┬──────┘
               │
               ▼
          Display
```

---

## License

This project is for educational and personal use.
