from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementConfig:
    target_fps: int = 30
    max_width: int = 760
    histogram_height: int = 125
    show_overlay: bool = True
    side_by_side: bool = True
    show_histogram: bool = True
    show_heatmap: bool = True
    subject_priority: bool = True
    suppress_highlights: bool = True
    low_light_threshold: float = 95.0
    low_contrast_threshold: float = 45.0
    shadow_level_threshold: float = 85.0
    shadow_ratio_threshold: float = 0.18
    dynamic_range_threshold: float = 105.0
    blur_threshold: float = 130.0
    high_motion_threshold: float = 16.0
    stabilize_luminance: bool = True

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / max(self.target_fps, 1)
