# Adaptive Video Enhancement Filter

An intelligent real-time video enhancement system for online communication environments. The project improves webcam/video quality dynamically instead of applying one fixed filter to every frame.

The system analyzes each frame for lighting, motion, sharpness, and estimated processing cost, then adapts enhancement strength so it can keep working on low-end devices and under unstable CPU conditions.

## Problem Statement

Online meetings often suffer from poor lighting, noisy camera output, low contrast, motion, and limited device performance. Static webcam filters usually apply the same processing regardless of frame quality or available compute, which can make video laggy or overprocessed.

This project proposes an adaptive balancing pipeline that:

- Improves visibility in dark areas without blowing out bright areas.
- Sharpens frames only when useful.
- Tracks motion and processing time to avoid expensive work when the device is under load.
- Keeps real-time communication usable by prioritizing responsiveness over heavy visual effects.

## Key Features

- **Frame quality analysis**: measures brightness, contrast, shadow level, highlight level, sharpness, motion, and processing latency.
- **Adaptive gamma correction**: lifts dark face regions while protecting already-bright areas.
- **CLAHE contrast enhancement**: improves local contrast on the luminance channel without over-amplifying lights.
- **Bilateral noise filtering**: reduces webcam noise while preserving edges.
- **Highlight suppression**: strongly compresses clipped light regions and their glow.
- **Unsharp masking**: restores midtone edge detail after denoising so the result stays clear.
- **Temporal tone smoothing**: stabilizes correction values and final luminance across frames to reduce light flicker or twinkling.
- **Performance-aware mode switching**: reduces processing strength when frames become expensive.
- **Webcam and video file support**: run on a live camera or a local video file.
- **Side-by-side comparison**: view original and enhanced video together for easy demonstration.
- **Live heatmap comparison**: view original and enhanced luminance heatmaps to inspect intensity equalization.
- **Live histogram graph**: compares original and enhanced intensity distribution in real time.
- **Metrics overlay**: displays live frame quality and mode information.

## Project Structure

```text
Adaptive_Video_Enhancement_-_Filter/
|-- README.md
|-- requirements.txt
|-- run.py
`-- src/
    |-- adaptive_enhancer/
    |   |-- __init__.py
    |   |-- analyzer.py
    |   |-- config.py
    |   |-- enhancer.py
    |   |-- pipeline.py
    |   `-- utils.py
    `-- tests/
        `-- test_analyzer.py
```

## How It Works

1. **Capture frame**  
   Frames are read from a webcam or video file using OpenCV.

2. **Analyze quality**  
   The analyzer estimates:
   - Brightness using grayscale mean intensity.
   - Contrast using grayscale standard deviation.
   - Shadow level using the 10th percentile of frame intensity.
   - Highlight level using the 90th percentile of frame intensity.
   - Sharpness using Laplacian variance.
   - Motion using frame difference against the previous frame.
   - System pressure using recent processing time.

3. **Choose adaptive settings**  
   The pipeline selects an enhancement profile:
   - `low_light`: stronger gamma, CLAHE, and highlight suppression for dark faces or mixed lighting.
   - `balanced`: moderate enhancement for normal webcam scenes.
   - `performance`: lighter processing only when processing time or motion stays high.

4. **Balance frame**  
   The enhancer applies:
   - Mild gray-world color balancing to reduce color cast.
   - Highlight suppression for strong lights and clipped glow.
   - Adaptive gamma correction for dark regions.
   - CLAHE on luminance for local contrast enhancement.
   - Bilateral filtering to reduce noise while keeping edges clear.
   - Unsharp masking to recover perceived sharpness.
   - Bright-region-focused temporal smoothing to reduce flickering lights without over-softening the face.

5. **Display output**  
   The original and enhanced frames are shown side by side with live metrics, mode status, live heatmaps, and a live intensity histogram graph. The heatmap uses cooler colors for darker regions and warmer colors for brighter regions. The histogram uses blue for the original frame and green for the enhanced frame.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Running The Project

Run with the default webcam. The output window shows the original video on the left and the enhanced video on the right:

```bash
python run.py
```

Run with a video file:

```bash
python run.py --source path/to/video.mp4
```

Use a different webcam index:

```bash
python run.py --source 1
```

Show only the enhanced video:

```bash
python run.py --enhanced-only
```

Hide the histogram graph:

```bash
python run.py --no-histogram
```

Hide the heatmap comparison:

```bash
python run.py --no-heatmap
```

Run in performance-first mode:

```bash
python run.py --target-fps 24 --max-width 640
```

Use a larger or smaller dashboard layout:

```bash
python run.py --max-width 700 --heatmap-height 120 --histogram-height 110
```

Press `q` in the video window to exit.

## Configuration

Important runtime options:

| Option | Description | Default |
| --- | --- | --- |
| `--source` | Webcam index or video path | `0` |
| `--target-fps` | Target frame rate used for performance decisions | `30` |
| `--max-width` | Resize each video panel before processing | `760` |
| `--heatmap-height` | Height of the heatmap comparison panel | `135` |
| `--histogram-height` | Height of the histogram panel | `125` |
| `--no-overlay` | Hide metrics overlay | Off |
| `--enhanced-only` | Hide the original/enhanced comparison and show only enhanced output | Off |
| `--no-histogram` | Hide the live intensity distribution graph | Off |
| `--no-heatmap` | Hide the original/enhanced heatmap comparison | Off |

## Why This Is Adaptive

The enhancement logic is not a single preset. It changes according to the observed frame and processing conditions:

- Dark regions receive controlled luminance correction.
- Bright regions are compressed instead of becoming brighter.
- Clipped background lights are pushed down more strongly than normal highlights.
- Frames with mixed lighting are balanced using shadow and highlight statistics.
- CLAHE strength increases when contrast is poor, but is masked away from bright lights.
- Bilateral filtering strength increases in darker/noisier scenes and decreases when motion is high.
- Mode switching chooses Low Light, Balanced, or Performance from scene brightness, motion, and processing time.
- Tone corrections and static bright luminance regions are smoothed over time so small frame-to-frame lighting changes do not cause visible twinkling.
- Unsharp masking adapts to measured sharpness and motion so the enhanced result remains clearer without becoming noisy.
- Low-detail frames receive controlled sharpening.
- High-motion scenes avoid heavy denoising to reduce latency.
- Slow processing automatically shifts toward lighter effects.

This makes the system more suitable for video calls, where smoothness and latency matter as much as visual clarity.

## Limitations

- It does not directly change network bitrate, but it can reduce processing load and resize frames to make downstream streaming easier.
- Real meeting-app integration would require virtual camera output or SDK integration, which can be added as a later module.

## Future Enhancements

- Add virtual camera output for Zoom, Teams, Meet, or OBS.
- Add automatic bitrate and resolution recommendations.
- Add GPU acceleration where available.
- Add a small dashboard for quality and performance trends.

## Testing

Run unit tests with:

```bash
pytest
```

The included test checks the quality analyzer on synthetic frames so the core measurement logic can be validated without a webcam.
