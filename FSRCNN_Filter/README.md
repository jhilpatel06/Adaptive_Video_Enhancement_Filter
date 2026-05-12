# FSRCNN Smart Video Filter

An intelligent real-time video filtering system that dynamically upscales facial regions using FSRCNN (Fast Super-Resolution Convolutional Neural Network) before applying stylistic effects. The project ensures that visual filters remain crisp and high-quality even when the subject is far from the camera or the video feed is low-resolution.

## Problem Statement

When users move away from their webcam or stream over a low-bandwidth connection, their face occupies fewer pixels. Applying stylistic filters (like Snapchat-style AR filters or color grading) directly to these low-resolution regions results in pixelated, noisy, and poor-quality effects. Running a heavy deep learning super-resolution model on the entire high-resolution frame is too slow for real-time applications.

This project proposes an adaptive, ROI-based (Region of Interest) pipeline that:

- Detects faces in real-time.
- Evaluates the size of the face relative to the total frame area.
- Selectively applies FSRCNN super-resolution *only* to the face if it falls below a certain size threshold.
- Applies the visual filter to the high-quality upscaled face.
- Seamlessly blends the filtered face back into the original frame without harsh edges.

## Key Features

- **Dynamic Super-Resolution**: Uses `FSRCNN_x4.pb` or `FSRCNN_x3.pb` to intelligently upscale low-resolution facial regions only when necessary (e.g., when the face area is < 5% of the frame).
- **Fast Face Detection**: Uses MediaPipe's robust BlazeFace/Face Landmarker models for high-speed tracking.
- **Modular Filter System**: Easily swap between various effects like Snapchat Prototype, Cyberpunk, Beautify, Cartoon, and Sepia.
- **Temporal Smoothing**: Applies Adaptive Exponential Moving Average (EMA) to facial landmarks to prevent jittering and provide smooth AR tracking.
- **Seamless Alpha Blending**: Uses feathered elliptical masks to blend the upscaled, filtered face naturally back into the unscaled environment.
- **Real-Time FSRCNN Comparison**: Includes a dedicated script (`realtime_fsrcnn.py`) to compare Ground Truth, Bicubic upscaling, and FSRCNN upscaling side-by-side with live PSNR (Peak Signal-to-Noise Ratio) and FPS metrics.

## Project Structure

```text
FSRCNN_Filter/
|-- README.md
|-- requirements.txt
|-- download_model.sh      # Script to download required FSRCNN models
|-- main.py                # Main application running the smart filter pipeline
|-- realtime_fsrcnn.py     # Standalone tool comparing FSRCNN vs Bicubic upscaling
|-- enhancer.py            # FSRCNN model wrapper for ROI upscaling
|-- face_detector.py       # MediaPipe face detection wrapper
`-- filters.py             # Collection of modular image filters (Snapchat, Cyberpunk, etc.)
```

## How It Works

1. **Capture & Detect**
   Frames are read from the webcam. MediaPipe identifies bounding boxes for all faces in the frame.

2. **Evaluate Area Threshold**
   The pipeline calculates the ratio of the face bounding box area to the total frame area. If the face is small (e.g., < 5%), it flags the face as needing enhancement.

3. **ROI Extraction & Upscaling**
   The bounding box is expanded slightly to capture the whole head (forehead to chin). If flagged for enhancement, this Region of Interest (ROI) is passed through the FSRCNN model to multiply its resolution.

4. **Apply Filter**
   The chosen filter (e.g., `Filters.apply_snapchat_prototype`) processes the ROI. Filters utilize smoothed facial landmarks to accurately draw AR elements like googly eyes or clown noses.

5. **Blend & Output**
   The filtered ROI is resized back to the original bounding box dimensions. A soft, feathered elliptical mask is generated to alpha-blend the filtered face back into the original frame, avoiding sharp rectangular borders. The result is displayed alongside the original frame for comparison.

## Installation

Ensure you have Python 3.10 or newer. It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Download Models**:
You need the FSRCNN TensorFlow PB models and the MediaPipe task file. You can use the provided shell script (if on Linux/macOS or Git Bash):
```bash
./download_model.sh
```
*(Ensure `FSRCNN_x3.pb`, `FSRCNN_x4.pb`, and `face_landmarker.task` are present in the directory).*

## Running The Project

### 1. Smart Filter Pipeline
Run the main pipeline that dynamically upscales and applies filters to faces. The output window shows the original video on the left and the smartly filtered video on the right.

```bash
python main.py
```
*Note: You can change the active filter by editing the `filter_func` argument in `main.py` (e.g., `Filters.apply_cyberpunk`, `Filters.apply_beautify`).*

### 2. FSRCNN vs Bicubic Real-Time Comparison
Run the standalone tool to analyze the raw performance and quality differences between basic Bicubic upscaling and FSRCNN neural network upscaling. This displays a live PSNR metric.

```bash
python realtime_fsrcnn.py
```

Press `q` in the video window to exit either application.

## Why This Is Adaptive

Instead of indiscriminately applying heavy processing to the entire 1080p or 720p frame (which would drop framerates to unusable levels), this system intelligently allocates compute power:

- **Distance-Aware Processing**: It only pays the performance cost of neural upscaling when the user is actually far away and pixelated.
- **Targeted Filtering**: It restricts stylistic pixel manipulation purely to the facial region.
- **Jitter Reduction**: It dynamically scales the smoothing strength of facial tracking points based on movement velocity.

## Future Enhancements

- Integrate the `Intensity_Optimizer`'s CLAHE and Gamma correction directly into the FSRCNN pipeline to handle both resolution and lighting simultaneously.
- Support for custom ONNX super-resolution models for better GPU acceleration via DirectML/TensorRT.
- Dynamic scale selection (switching between x2, x3, or x4 based on the exact face size).
