from __future__ import annotations

import time

import cv2

from .analyzer import FrameMetrics, FrameQualityAnalyzer
from .config import EnhancementConfig
from .enhancer import AdaptiveFrameEnhancer
from .utils import (
    compose_side_by_side,
    compose_with_heatmap,
    compose_with_histogram,
    draw_overlay,
    resize_to_max_width,
)


class VideoEnhancementPipeline:
    def __init__(self, config: EnhancementConfig) -> None:
        self.config = config
        self.analyzer = FrameQualityAnalyzer()
        self.enhancer = AdaptiveFrameEnhancer(config)
        self._last_processing_ms = 0.0
        self._mode = "balanced"
        self._overload_frames = 0
        self._low_light_frames = 0

    def run(self, source: int | str = 0) -> None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        cv2.namedWindow("Adaptive Video Enhancement", cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = resize_to_max_width(frame, self.config.max_width)
                start = time.perf_counter()

                metrics = self.analyzer.analyze(frame, self._last_processing_ms)
                mode = self._select_mode(metrics)
                output = self.enhancer.enhance(frame, metrics, mode)

                self._last_processing_ms = (time.perf_counter() - start) * 1000.0
                metrics = FrameMetrics(
                    brightness=metrics.brightness,
                    contrast=metrics.contrast,
                    shadow_level=metrics.shadow_level,
                    highlight_level=metrics.highlight_level,
                    shadow_ratio=metrics.shadow_ratio,
                    highlight_ratio=metrics.highlight_ratio,
                    blur_score=metrics.blur_score,
                    motion_score=metrics.motion_score,
                    processing_ms=self._last_processing_ms,
                )

                display = output
                if self.config.side_by_side:
                    display = compose_side_by_side(frame, output)

                if self.config.show_overlay:
                    draw_overlay(display, metrics, mode)

                if self.config.show_heatmap:
                    display = compose_with_heatmap(display, frame, output, frame.shape[0])

                if self.config.show_histogram:
                    display = compose_with_histogram(display, frame, output, self.config.histogram_height)

                # Scale down if it's too tall for most screens
                max_display_height = 820
                if display.shape[0] > max_display_height:
                    scale = max_display_height / display.shape[0]
                    display = cv2.resize(
                        display, 
                        (int(display.shape[1] * scale), max_display_height), 
                        interpolation=cv2.INTER_AREA
                    )

                cv2.imshow("Adaptive Video Enhancement", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _select_mode(self, metrics: FrameMetrics) -> str:
        overloaded = (
            metrics.processing_ms > self.config.frame_budget_ms * 2.4
            or metrics.motion_score > self.config.high_motion_threshold * 2.2
        )
        low_light = (
            metrics.shadow_level < self.config.shadow_level_threshold
            or metrics.shadow_ratio > self.config.shadow_ratio_threshold
            or metrics.brightness < self.config.low_light_threshold
        )
        self._overload_frames = self._overload_frames + 1 if overloaded else max(0, self._overload_frames - 1)
        self._low_light_frames = self._low_light_frames + 1 if low_light else max(0, self._low_light_frames - 1)

        if self._overload_frames >= 12:
            self._mode = "performance"
        elif self._low_light_frames >= 3:
            self._mode = "low_light"
        elif self._overload_frames == 0 and self._low_light_frames == 0:
            self._mode = "balanced"

        return self._mode
