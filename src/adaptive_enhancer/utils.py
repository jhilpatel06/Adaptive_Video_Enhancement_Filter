from __future__ import annotations

import cv2
import numpy as np

from .analyzer import FrameMetrics


def resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_size = (max_width, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def compose_side_by_side(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    if original.shape[:2] != enhanced.shape[:2]:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)

    left = original.copy()
    right = enhanced.copy()
    _draw_label(left, "Original")
    _draw_label(right, "Enhanced")
    return np.hstack((left, right))


def compose_with_histogram(video_panel: np.ndarray, original: np.ndarray, enhanced: np.ndarray, height: int) -> np.ndarray:
    histogram = draw_luminance_histogram(original, enhanced, width=video_panel.shape[1], height=height)
    return np.vstack((video_panel, histogram))


def compose_with_heatmap(video_panel: np.ndarray, original: np.ndarray, enhanced: np.ndarray, height: int) -> np.ndarray:
    heatmap = draw_luminance_heatmap_pair(original, enhanced, width=video_panel.shape[1], height=height)
    return np.vstack((video_panel, heatmap))


def draw_luminance_heatmap_pair(original: np.ndarray, enhanced: np.ndarray, width: int, height: int) -> np.ndarray:
    half_width = width // 2
    left = _luminance_heatmap(original, half_width, height)
    right = _luminance_heatmap(enhanced, width - half_width, height)
    _draw_label(left, "Original Heatmap")
    _draw_label(right, "Enhanced Heatmap")
    return np.hstack((left, right))


def _luminance_heatmap(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_AREA)
    _draw_heatmap_scale(heat)
    return heat


def _draw_heatmap_scale(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    bar_width = 14
    bar_height = max(58, height - 52)
    x0 = width - bar_width - 14
    y0 = 38
    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8)[:, None]
    gradient = np.repeat(gradient, bar_width, axis=1)
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    frame[y0 : y0 + bar_height, x0 : x0 + bar_width] = bar
    cv2.rectangle(frame, (x0, y0), (x0 + bar_width, y0 + bar_height), (245, 245, 245), 1)
    cv2.putText(frame, "bright", (x0 - 48, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, "dark", (x0 - 35, y0 + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)


def draw_luminance_histogram(original: np.ndarray, enhanced: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    plot_top = 24
    plot_bottom = height - 24
    plot_left = 56
    plot_right = width - 22

    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (52, 52, 52), 1)
    for ratio in (0.25, 0.5, 0.75):
        x = int(plot_left + (plot_right - plot_left) * ratio)
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), (40, 40, 40), 1)

    original_hist = _normalized_luminance_histogram(original, plot_bottom - plot_top)
    enhanced_hist = _normalized_luminance_histogram(enhanced, plot_bottom - plot_top)
    _draw_histogram_curve(canvas, original_hist, plot_left, plot_top, plot_right, plot_bottom, (90, 170, 255))
    _draw_histogram_curve(canvas, enhanced_hist, plot_left, plot_top, plot_right, plot_bottom, (90, 235, 130))

    cv2.putText(canvas, "Live intensity distribution", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "dark", (12, plot_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "bright", (plot_right - 52, plot_bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Original", (plot_left, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 170, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Enhanced", (plot_left + 100, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 235, 130), 1, cv2.LINE_AA)
    return canvas


def _normalized_luminance_histogram(frame: np.ndarray, max_height: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = np.log1p(hist)
    peak = float(np.max(hist))
    if peak <= 0:
        return np.zeros_like(hist)
    return (hist / peak) * max_height


def _draw_histogram_curve(
    canvas: np.ndarray,
    hist: np.ndarray,
    plot_left: int,
    plot_top: int,
    plot_right: int,
    plot_bottom: int,
    color: tuple[int, int, int],
) -> None:
    plot_width = plot_right - plot_left
    points = []
    for index, value in enumerate(hist):
        x = int(plot_left + (index / 255.0) * plot_width)
        y = int(plot_bottom - value)
        y = max(plot_top, min(plot_bottom, y))
        points.append((x, y))

    for start, end in zip(points, points[1:]):
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)


def _draw_label(frame: np.ndarray, label: str) -> None:
    x, y = 12, 34
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)

    cv2.rectangle(
        frame,
        (0, 0),
        (text_width + 28, text_height + baseline + 22),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, metrics: FrameMetrics, mode: str) -> None:
    rows = [
        f"mode: {mode}",
        f"brightness: {metrics.brightness:5.1f}",
        f"contrast: {metrics.contrast:5.1f}",
        f"shadow p10: {metrics.shadow_level:5.1f}",
        f"shadow %: {metrics.shadow_ratio * 100:5.1f}",
        f"highlight %: {metrics.highlight_ratio * 100:5.1f}",
        f"sharpness: {metrics.blur_score:6.1f}",
        f"motion: {metrics.motion_score:5.1f}",
        f"processing: {metrics.processing_ms:5.1f} ms",
    ]

    x, y = 12, 24
    line_height = 24
    panel_width = 250
    panel_height = line_height * len(rows) + 14

    cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.82, frame, 0.18, 0, frame)

    for index, text in enumerate(rows):
        cv2.putText(
            frame,
            text,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
