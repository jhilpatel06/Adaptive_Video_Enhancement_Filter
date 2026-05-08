import numpy as np

from src.adaptive_enhancer.analyzer import FrameQualityAnalyzer


def test_analyzer_reports_expected_synthetic_frame_values():
    analyzer = FrameQualityAnalyzer()
    dark_frame = np.zeros((80, 120, 3), dtype=np.uint8)
    bright_frame = np.full((80, 120, 3), 220, dtype=np.uint8)

    dark_metrics = analyzer.analyze(dark_frame)
    bright_metrics = analyzer.analyze(bright_frame)

    assert dark_metrics.brightness == 0
    assert bright_metrics.brightness == 220
    assert bright_metrics.motion_score > 0
