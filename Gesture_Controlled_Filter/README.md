# Gesture Controlled Cool/Warm Filter

An interactive real-time video filter that adjusts the color temperature of your webcam feed based on hand gestures. 

This project uses computer vision and hand-tracking to seamlessly change the "mood" or temperature of the video simply by changing the distance between your thumb and index finger.

## Problem Statement

Traditional webcam filters rely on static sliders or preset configurations to change the look of the video. To adjust the color temperature or tone of a stream, a user typically has to navigate menus and tweak UI sliders, which breaks immersion and is not user-friendly during live communication.

This project proposes an intuitive, touchless interaction model that:
- Uses physical gestures to control software parameters in real time.
- Instantly visualizes the result alongside the original feed.
- Provides immediate visual feedback through a dynamic UI scale.

## Key Features

- **Live Hand Tracking**: Detects and tracks the user's hand in real time using Mediapipe.
- **Gesture Mapping**: Translates the physical distance between the thumb and index finger into a value between -100 (Cool) and +100 (Warm).
- **Dynamic Color Temperature Filter**: 
  - Automatically shifts the video toward a cooler, blue-heavy tone when fingers are close together.
  - Automatically shifts the video toward a warmer, red-heavy tone when fingers are spread apart.
- **Side-by-side Comparison**: Displays the original webcam feed (with hand-tracking overlay) right next to the filtered output for instant visual comparison.
- **Interactive Visual Scale**: Renders a dynamic, thermometer-like UI element on the screen that fills with blue or red depending on the current temperature.

## Project Structure

```text
Warm_Filter/
|-- README.md
|-- requirements.txt
`-- gesture_filter.py
```

## How It Works

1. **Capture Frame**  
   Frames are read from the default webcam using OpenCV and flipped horizontally to act like a mirror (selfie-view).

2. **Track Gestures**  
   The frame is processed by the Mediapipe Hands model. The model identifies 21 3D landmarks of the hand. 
   We specifically extract:
   - Landmark 4: Thumb Tip
   - Landmark 8: Index Finger Tip

3. **Calculate Temperature**  
   The Euclidean pixel distance between the thumb tip and index finger tip is calculated. This distance (ranging from ~20 to 200 pixels) is mapped linearly to a temperature value of -100 to 100.

4. **Apply Filter**  
   - **If Temperature > 0**: The red color channel of the frame is increased while the blue channel is decreased, creating a warm/orange tint.
   - **If Temperature < 0**: The blue color channel is increased while the red channel is decreased, creating a cool/blue tint.
   - The pixel intensities are clipped to the valid 0-255 range to prevent overflow artifacts.

5. **Display Output**  
   The system draws a visual line between your fingers and displays the live distance on the original frame. It draws a thermometer bar on the filtered frame. Finally, both frames are stacked horizontally and displayed in a single window.

## Installation

Ensure you have Python 3.10 or newer installed. 
It is recommended to use a virtual environment to manage dependencies cleanly.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

On Linux/macOS, activate the virtual environment with:
```bash
source .venv/bin/activate
```

## Running The Project

Run the application with the following command:

```bash
python gesture_filter.py
```

Press **`q`** in the video window to safely close the application.

## How to Use the Gestures

1. Open the application.
2. Raise your hand so it is clearly visible in the camera frame.
3. **Pinch (Cool Filter)**: Bring your thumb and index finger close together. You will see the visual scale fill with blue downwards, and the right side of the screen will become cooler.
4. **Spread (Warm Filter)**: Spread your thumb and index finger far apart. You will see the visual scale fill with red upwards, and the right side of the screen will become warmer.

## Limitations

- The hand-tracking relies on decent lighting. If the room is completely dark, the model may fail to identify your hand.
- The filter is currently a simple RGB channel manipulation rather than a complex color space conversion (like LAB), so extreme temperature values might look slightly artificial.
- Currently supports tracking one hand at a time.

## Future Enhancements

- Convert image to LAB color space for more natural-looking temperature adjustments.
- Support multi-hand gestures (e.g., left hand controls brightness, right hand controls temperature).
- Add a smoothing filter (exponential moving average) to the distance calculation to prevent the color temperature from flickering if the hand tracking jitters slightly.
