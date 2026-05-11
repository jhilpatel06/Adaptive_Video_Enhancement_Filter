from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    brightness: float
    contrast: float
    shadow_level: float
    highlight_level: float
    shadow_ratio: float
    highlight_ratio: float
    blur_score: float
    motion_score: float
    processing_ms: float


class FrameQualityAnalyzer:
    """Computes lightweight quality metrics suitable for real-time use."""

    def __init__(self) -> None:
        self._previous_gray: np.ndarray | None = None

    def analyze(self, frame: np.ndarray, processing_ms: float = 0.0) -> FrameMetrics:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        shadow_level = float(np.percentile(gray, 10))
        highlight_level = float(np.percentile(gray, 90))
        shadow_ratio = float(np.mean(gray < 75))
        highlight_ratio = float(np.mean(gray > 225))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        motion_score = self._motion_score(gray)

        self._previous_gray = gray

        return FrameMetrics(
            brightness=brightness,
            contrast=contrast,
            shadow_level=shadow_level,
            highlight_level=highlight_level,
            shadow_ratio=shadow_ratio,
            highlight_ratio=highlight_ratio,
            blur_score=blur_score,
            motion_score=motion_score,
            processing_ms=processing_ms,
        )

    def _motion_score(self, gray: np.ndarray) -> float:
        if self._previous_gray is None:
            return 0.0

        if self._previous_gray.shape != gray.shape:
            return 0.0

        diff = cv2.absdiff(gray, self._previous_gray)
        return float(np.mean(diff))
