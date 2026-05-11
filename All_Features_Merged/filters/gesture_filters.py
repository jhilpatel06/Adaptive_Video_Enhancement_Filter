import cv2
import numpy as np


def apply_temperature(frame, temperature, face_id=0):
    """
    Gesture-controlled cool/warm color temperature filter.

    Shifts the image toward blue (cool) when temperature < 0,
    and toward red (warm) when temperature > 0.

    Args:
        frame:       BGR image.
        temperature: float in [-100, 100].
        face_id:     unused – included for pipeline signature compatibility.

    Returns:
        Filtered BGR image.
    """
    filtered = frame.copy().astype(np.float32)

    if temperature > 0:
        r_change = (temperature / 100.0) * 60
        b_change = (temperature / 100.0) * -60
        filtered[:, :, 2] += r_change   # Red channel
        filtered[:, :, 0] += b_change   # Blue channel
    elif temperature < 0:
        r_change = (abs(temperature) / 100.0) * -60
        b_change = (abs(temperature) / 100.0) * 60
        filtered[:, :, 2] += r_change
        filtered[:, :, 0] += b_change

    return np.clip(filtered, 0, 255).astype(np.uint8)


def draw_thermometer(frame, temperature):
    """
    Draws a visual thermometer-like scale on the frame showing the
    current temperature value.

    Args:
        frame:       BGR image (modified in-place).
        temperature: float in [-100, 100].
    """
    bar_x, bar_y = 50, 100
    bar_w, bar_h = 30, 200

    # Background bar (gray)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)

    # Colored fill based on temperature
    fill_y = int(np.interp(temperature, [-100, 100], [bar_y + bar_h, bar_y]))

    if temperature > 0:
        bar_color = (0, 0, 255)    # Red for Warm
    elif temperature < 0:
        bar_color = (255, 0, 0)    # Blue for Cool
    else:
        bar_color = (0, 255, 0)    # Green for Neutral

    cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), 2)

    # Labels
    cv2.putText(frame, "Warm", (bar_x - 15, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Cool", (bar_x - 10, bar_y + bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"{int(temperature)}", (bar_x + 40, fill_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
