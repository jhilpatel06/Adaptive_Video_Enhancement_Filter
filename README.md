# Adaptive Video Enhancement & Smart Filter System

A comprehensive, AI-powered real-time video processing pipeline designed to dynamically improve webcam quality and apply intelligent, adaptive visual filters. 

Traditional webcam filters are often static: they apply the same heavy processing regardless of lighting conditions, device performance, or how far the user is from the camera. This project solves these issues by breaking down video enhancement into a multi-layered, adaptive architecture.

---

## The Core Problem

Online communication often suffers from a combination of poor environments and rigid software:
1. **Unpredictable Lighting:** Rooms are either too dark, causing noisy shadows, or have harsh backlighting that blows out the face. Static brightness sliders fail when lighting changes or varies across the frame.
2. **Resolution Loss:** When a user moves away from the camera, their face occupies fewer pixels. Applying complex stylistic AR filters to these low-resolution areas results in pixelated, noisy, and poor-quality effects.
3. **Rigid Interfaces:** Adjusting camera settings (like color temperature) usually requires clicking through menus, breaking immersion.
4. **Performance Constraints:** Applying deep learning super-resolution or heavy denoising to an entire 1080p frame drops the framerate to unusable levels on standard CPUs.

---

## Project Architecture & Modules

This repository is structured into modular components, culminating in a final merged pipeline.

### 1. Intensity Optimizer (`Intensity_Optimizer/`)
This module acts as the foundation, ensuring the base video feed is clean and well-lit before any stylistic filters are applied. It continuously analyzes the frame for brightness, contrast, motion, and system pressure.

*   **Adaptive Gamma Correction:** Lifts dark facial regions while protecting already-bright areas from blowing out.
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Improves local contrast specifically on the luminance channel.
*   **Highlight Suppression:** Strongly compresses clipped light regions and their glow.
*   **Temporal Tone Smoothing:** Stabilizes correction values across frames to reduce light flicker or "twinkling".
*   **Performance-Aware:** Automatically shifts processing strength (e.g., reducing bilateral filtering) if the framerate drops or motion is too high.

### 2. FSRCNN Smart Filter (`FSRCNN_Filter/`)
This module handles spatial resolution and stylization. It ensures that visual filters remain crisp even when the subject is far from the camera.

*   **Distance-Aware Processing:** Uses MediaPipe Face Detection to calculate the ratio of the face area to the frame area.
*   **Targeted Super-Resolution:** If the face is small (e.g., < 5% of the frame), it extracts the Region of Interest (ROI) and runs it through an **FSRCNN** (Fast Super-Resolution Convolutional Neural Network) model to multiply its resolution.
*   **Stylistic Filters:** Applies filters (Snapchat AR prototypes, Cyberpunk color grading, Beautify, etc.) to the high-quality upscaled face.
*   **Seamless Alpha Blending:** Uses feathered elliptical masks to blend the upscaled, filtered face naturally back into the unscaled background.

### 3. Gesture-Controlled Filter (`Gesture_Controlled_Filter/`)
This module introduces an interactive, touchless interface for controlling the video's mood.

*   **Live Hand Tracking:** Utilizes MediaPipe Hands to detect 3D landmarks.
*   **Distance Mapping:** Calculates the physical distance between the thumb and index finger.
*   **Dynamic Color Temperature:** 
    *   **Pinch:** Shifts the video to a cooler, blue-heavy tone.
    *   **Spread:** Shifts the video to a warmer, red-heavy orange tone.
*   **Visual Feedback:** Renders an interactive thermometer-style UI scale directly on the screen.

### 4. The Unified Pipeline (`All_Features_Merged/`)
This is the culmination of the project, merging all three modules into a single, cohesive architecture. 

**How the Unified Pipeline Works:**
1.  **Phase 1 (Optimization):** The raw frame passes through the Intensity Optimizer to fix lighting and reduce noise.
2.  **Phase 2 (Detection):** The optimized frame is analyzed simultaneously for Hands (Gesture) and Faces.
3.  **Phase 3 (Resolution Scaling):**
    *   **Downscaling:** The facial Region of Interest (ROI) is extracted and downscaled (to simulate a low-resolution feed or standardize input size for performance).
    *   **Upscaling:** The low-resolution ROI is passed through the FSRCNN neural network to reconstruct lost details and output a crisp, high-resolution face.
4.  **Phase 4 (Stylization & Blending):** 
    *   If a gesture is detected, the global color temperature shifts.
    *   The active stylistic filter (e.g., Cartoon, Sepia, Snapchat) is applied specifically to the high-resolution face, which is then seamlessly alpha-blended back into the original frame.
5.  **Phase 5 (Overlays):** Metrics (FPS, Mode), luminance heatmaps, and intensity histograms are drawn on top.

---

## In-Depth Image Processing Techniques

The system utilizes several classical and modern computer vision techniques to achieve its results. These are particularly important for understanding the "why" behind the visual improvements:

1. **Gamma Correction (Non-Linear Luminance Adjustment)**
   * **How it works:** Applies a power-law transformation ($V_{out} = V_{in}^\gamma$) to pixel intensities.
   * **Why it helps:** Unlike linear brightness scaling which easily blows out highlights, gamma correction curves the mapping to aggressively stretch dark shadow values (making them visible) while gently compressing already bright values.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   * **How it works:** Instead of calculating one global histogram for the entire image, CLAHE divides the image into an 8x8 grid of "tiles". It equalizes the histogram of each tile independently. Crucially, it clips the histogram at a certain threshold to prevent noise amplification in uniform areas (like flat walls).
   * **Why it helps:** It brings out hidden local textures and details that global contrast adjustments miss. By performing this exclusively on the **L-channel** (Lightness) of the LAB color space, it enhances contrast without distorting the original colors.
3. **Bilateral Filtering**
   * **How it works:** A non-linear, edge-preserving smoothing filter. Standard Gaussian blur replaces a pixel with the average of its spatial neighbors. Bilateral filtering also considers *intensity* differences, meaning it only averages pixels that are both close together *and* similar in color/brightness.
   * **Why it helps:** It aggressively removes webcam sensor noise (grain) in flat regions (like skin or shadows) but completely halts blurring when it detects an edge (like an eye or the contour of a face), preserving sharpness.
4. **Unsharp Masking**
   * **How it works:** It creates a slightly blurred version of the image, subtracts it from the original to isolate high-frequency edge information (the "mask"), and adds that mask back to the original image.
   * **Why it helps:** Because Bilateral Filtering can sometimes make faces look slightly "plastic," unsharp masking restores perceived midtone sharpness and texture, ensuring the final output remains crisp.
5. **Alpha Blending with Elliptical Feathering**
   * **How it works:** When blending the FSRCNN-upscaled face back onto the original frame, a hard rectangular cut would look jarring. The system generates an elliptical mask covering the face, applies a heavy Gaussian blur to the mask, and uses it as an alpha channel ($Output = Filtered \times Alpha + Original \times (1 - Alpha)$).
   * **Why it helps:** It ensures a photographically seamless transition between the high-resolution, stylized facial region and the unprocessed background environment.
6. **Exponential Moving Average (EMA) Temporal Smoothing**
   * **How it works:** Applies a low-pass filter across time. For any data point $P$ at time $t$: $P_t = \alpha \cdot Current + (1-\alpha) \cdot P_{t-1}$. The system dynamically adjusts $\alpha$ based on movement velocity.
   * **Why it helps:** It prevents AR filter jitter (e.g., googly eyes vibrating) and stops the lighting enhancement from flickering rapidly if the webcam auto-exposure shifts slightly.

---

## Visual Analytics & Plots

To validate the image processing pipeline and provide immediate feedback, the system generates real-time telemetry and plots:

1. **Real-Time Intensity Histogram (Line Graph)**
   * **What it shows:** Plots the distribution of pixel brightness values (0 = pure black, 255 = pure white). 
   * **Observation:** The original feed (Blue line) often shows heavy clustering at the dark end (left side). The enhanced feed (Green line) visually demonstrates the histogram being stretched and centralized, proving that shadows have been lifted and overall contrast is maximized without clipping.
2. **Luminance Heatmap Comparison**
   * **What it shows:** Converts the grayscale intensity of the frame into a false-color Jet colormap (Dark/Cold = Blue, Midtones = Green/Yellow, Bright/Hot = Red).
   * **Observation:** Side-by-side heatmaps allow developers to quickly see "hotspots" (blown-out windows) or "cold spots" (underexposed faces). The enhanced heatmap usually reveals a much more even, green/yellow thermal distribution across the subject's face.
3. **PSNR Metric (Peak Signal-to-Noise Ratio)**
   * **What it shows:** Calculated in the standalone FSRCNN comparison tool (`realtime_fsrcnn.py`), PSNR measures the logarithmic decibel (dB) difference between a "Ground Truth" high-res frame and the upscaled frame.
   * **Observation:** Objectively proves the superiority of the Neural Network over traditional math. FSRCNN consistently yields higher PSNR values compared to standard Bicubic Interpolation, correlating with visibly sharper edges and fewer artifacts.
4. **Gesture Thermometer UI**
   * **What it shows:** An on-screen UI element that maps the physical Euclidean pixel distance between the thumb and index finger to an arbitrary bounded scale (-100 to +100).
   * **Observation:** Provides crucial UX feedback, letting the user know exactly where their hand is registering on the color temperature spectrum before the visual extreme is reached.

---

## Installation & Setup

To run the complete merged pipeline or any individual module, ensure you have Python 3.10+ installed.

### 1. Clone & Environment setup
```bash
git clone https://github.com/jhilpatel06/Adaptive_Video_Enhancement_-_Filter.git
cd Adaptive_Video_Enhancement_-_Filter

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
# You can install requirements from the merged folder
pip install -r All_Features_Merged/requirements.txt
```

### 3. Download Required AI Models
The unified pipeline requires specific model binaries that are not included in the git repository due to size. Place these in the `All_Features_Merged/` directory:

1.  `blaze_face_full_range.tflite` (MediaPipe Face Detection)
2.  `face_landmarker.task` (MediaPipe Face Landmarker)
3.  `hand_landmarker.task` (MediaPipe Hand Landmarker)
4.  `FSRCNN_x4.pb` or `FSRCNN_x3.pb` (OpenCV Super Resolution Models)

*(Note: The `FSRCNN_Filter/` folder includes a `download_model.sh` script that can assist in fetching the FSRCNN models).*

---

## Usage (Unified Pipeline)

Navigate to the `All_Features_Merged` directory and run the main entry point:

```bash
cd All_Features_Merged
python main.py
```

### Controls:
*   **Keyboard `1` – `9`**: Switch between different stylistic filters (Sepia, Cyberpunk, Snapchat Prototype, etc.).
*   **Keyboard `n` / `p`**: Cycle to the Next or Previous filter.
*   **Keyboard `m`**: Toggle the metrics overlay (Brightness, Contrast, FPS).
*   **Keyboard `h`**: Toggle the luminance heatmap comparison panel.
*   **Keyboard `g`**: Toggle the intensity histogram graph.
*   **Gestures**: Raise your hand into the frame. Pinch fingers together to cool the image, spread them apart to warm it.
*   **Keyboard `q`**: Quit the application.

---

## Future Enhancements

*   **Virtual Camera Integration:** Route the processed output directly into Zoom, Teams, or Google Meet via OBS or virtual camera SDKs.
*   **Hardware Acceleration:** Transition the FSRCNN model inference to ONNX/TensorRT for massive GPU-based performance gains.
*   **Expanded Gestures:** Support multi-hand controls (e.g., Left hand for brightness, Right hand for color temperature).
*   **Advanced Color Spaces:** Perform gesture color temperature shifts in the LAB color space rather than RGB for more natural-looking skin tones during extreme adjustments.
