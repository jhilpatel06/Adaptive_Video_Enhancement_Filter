from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analyzer import FrameMetrics
from .config import EnhancementConfig


@dataclass(frozen=True)
class EnhancementProfile:
    gamma: float
    clahe_clip: float
    clahe_blend: float
    bilateral_diameter: int
    bilateral_sigma: float
    sharpen_amount: float
    highlight_strength: float


class AdaptiveFrameEnhancer:
    def __init__(self, config: EnhancementConfig) -> None:
        self.config = config
        self._state: dict[str, float] = {}
        self._previous_luminance: np.ndarray | None = None

    def enhance(self, frame: np.ndarray, metrics: FrameMetrics, mode: str) -> np.ndarray:
        profile = self._profile_for(metrics, mode)

        enhanced = self._gray_world_balance(frame, strength=0.12 if mode == "performance" else 0.16)
        enhanced = self._suppress_highlights(enhanced, metrics, profile)
        enhanced = self._apply_adaptive_gamma(enhanced, profile)
        enhanced = self._apply_clahe(enhanced, profile)
        enhanced = self._apply_bilateral_filter(enhanced, profile)
        enhanced = self._apply_unsharp_mask(enhanced, metrics, profile)

        if self.config.stabilize_luminance:
            enhanced = self._stabilize_bright_luminance(enhanced, metrics)

        return enhanced

    def _profile_for(self, metrics: FrameMetrics, mode: str) -> EnhancementProfile:
        shadow_need = np.clip((88.0 - metrics.shadow_level) / 75.0, 0.0, 1.0)
        shadow_need = max(shadow_need, np.clip((metrics.shadow_ratio - 0.12) / 0.30, 0.0, 1.0))
        shadow_need = self._smooth_value("shadow_need", float(shadow_need), alpha=0.06)

        highlight_need = np.clip((metrics.highlight_ratio - 0.015) / 0.18, 0.0, 1.0)
        highlight_need = max(highlight_need, np.clip((metrics.highlight_level - 178.0) / 62.0, 0.0, 1.0))
        highlight_need = self._smooth_value("highlight_need", float(highlight_need), alpha=0.05)

        motion = np.clip(metrics.motion_score / max(self.config.high_motion_threshold, 1.0), 0.0, 1.0)
        detail_need = np.clip((240.0 - metrics.blur_score) / 220.0, 0.0, 1.0)

        mode_scale = {
            "low_light": 1.0,
            "balanced": 0.86,
            "performance": 0.72,
        }[mode]

        gamma = 1.0 - (0.26 + 0.30 * mode_scale) * shadow_need
        gamma = float(np.clip(gamma, 0.46, 1.0))
        gamma = self._smooth_value("gamma", gamma, alpha=0.06)

        clahe_clip = 1.55 + 1.05 * shadow_need * mode_scale
        clahe_clip *= 1.0 - 0.18 * motion
        clahe_clip = self._smooth_value("clahe_clip", float(np.clip(clahe_clip, 1.35, 2.45)), alpha=0.05)

        clahe_blend = 0.38 + 0.24 * shadow_need * mode_scale
        clahe_blend *= 1.0 - 0.15 * motion
        clahe_blend = self._smooth_value("clahe_blend", float(np.clip(clahe_blend, 0.30, 0.58)), alpha=0.05)

        bilateral_diameter = 5 if mode == "performance" else 7
        bilateral_sigma = 28.0 + 30.0 * shadow_need
        bilateral_sigma *= 1.0 - 0.45 * motion

        sharpen_amount = 0.48 + 0.46 * detail_need
        sharpen_amount *= 1.0 - 0.25 * motion
        if mode == "performance":
            sharpen_amount *= 0.92

        highlight_strength = 0.82 + 0.17 * highlight_need
        highlight_strength = self._smooth_value(
            "highlight_strength",
            float(np.clip(highlight_strength, 0.78, 0.99)),
            alpha=0.05,
        )

        return EnhancementProfile(
            gamma=gamma,
            clahe_clip=clahe_clip,
            clahe_blend=clahe_blend,
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma=float(np.clip(bilateral_sigma, 18.0, 58.0)),
            sharpen_amount=float(np.clip(sharpen_amount, 0.34, 0.92)),
            highlight_strength=highlight_strength,
        )

    def _gray_world_balance(self, frame: np.ndarray, strength: float) -> np.ndarray:
        channels = cv2.split(frame.astype(np.float32))
        means = [float(np.mean(channel)) for channel in channels]
        gray_mean = sum(means) / len(means)
        balanced = [
            np.clip(channel * np.clip(gray_mean / max(mean, 1.0), 0.94, 1.06), 0, 255)
            for channel, mean in zip(channels, means)
        ]
        balanced_frame = cv2.merge(balanced)
        blended = frame.astype(np.float32) * (1.0 - strength) + balanced_frame * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _suppress_highlights(
        self,
        frame: np.ndarray,
        metrics: FrameMetrics,
        profile: EnhancementProfile,
    ) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_float = l_channel.astype(np.float32)

        hot_mask = np.clip((l_float - 152.0) / 64.0, 0.0, 1.0) ** 0.82
        extreme_mask = np.clip((l_float - 190.0) / 30.0, 0.0, 1.0)
        hot_mask = np.maximum(hot_mask, extreme_mask)
        hot_mask = cv2.GaussianBlur(hot_mask, (0, 0), sigmaX=10.5, sigmaY=10.5)

        compression_target = 140.0 if metrics.highlight_ratio > 0.04 else 152.0
        compressed = compression_target + (l_float - compression_target) * 0.07
        compressed = np.minimum(compressed, 174.0)

        blend = np.clip(hot_mask * profile.highlight_strength, 0.0, 1.0)
        l_suppressed = l_float * (1.0 - blend) + compressed * blend

        chroma_blend = np.clip(blend * 0.68, 0.0, 1.0)
        a_float = a_channel.astype(np.float32) * (1.0 - chroma_blend) + 128.0 * chroma_blend
        b_float = b_channel.astype(np.float32) * (1.0 - chroma_blend) + 128.0 * chroma_blend

        return cv2.cvtColor(
            cv2.merge(
                (
                    np.clip(l_suppressed, 0, 235).astype(np.uint8),
                    np.clip(a_float, 0, 255).astype(np.uint8),
                    np.clip(b_float, 0, 255).astype(np.uint8),
                )
            ),
            cv2.COLOR_LAB2BGR,
        )

    def _apply_adaptive_gamma(self, frame: np.ndarray, profile: EnhancementProfile) -> np.ndarray:
        gamma = max(profile.gamma, 0.05)
        table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.float32)
        table = np.clip(table, 0, 255).astype(np.uint8)
        corrected = cv2.LUT(frame, table)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shadow_mask = np.clip((155.0 - gray) / 155.0, 0.0, 1.0)[:, :, None]
        highlight_guard = 1.0 - np.clip((gray - 170.0) / 55.0, 0.0, 1.0)[:, :, None]
        blend = shadow_mask * highlight_guard
        mixed = frame.astype(np.float32) * (1.0 - blend) + corrected.astype(np.float32) * blend
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def _apply_clahe(self, frame: np.ndarray, profile: EnhancementProfile) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_float = l_channel.astype(np.float32)

        clahe = cv2.createCLAHE(clipLimit=profile.clahe_clip, tileGridSize=(8, 8))
        equalized = clahe.apply(l_channel).astype(np.float32)

        shadow_mid_mask = np.clip((205.0 - l_float) / 185.0, 0.0, 1.0)
        highlight_guard = 1.0 - np.clip((l_float - 170.0) / 50.0, 0.0, 1.0)
        blend = profile.clahe_blend * shadow_mid_mask * highlight_guard
        l_equalized = l_float * (1.0 - blend) + equalized * blend

        return cv2.cvtColor(
            cv2.merge((np.clip(l_equalized, 0, 238).astype(np.uint8), a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    def _apply_bilateral_filter(self, frame: np.ndarray, profile: EnhancementProfile) -> np.ndarray:
        filtered = cv2.bilateralFilter(
            frame,
            profile.bilateral_diameter,
            profile.bilateral_sigma,
            profile.bilateral_sigma,
        )
        return cv2.addWeighted(frame, 0.56, filtered, 0.44, 0)

    def _apply_unsharp_mask(
        self,
        frame: np.ndarray,
        metrics: FrameMetrics,
        profile: EnhancementProfile,
    ) -> np.ndarray:
        blur_sigma = 1.05
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        sharpened = cv2.addWeighted(frame, 1.0 + profile.sharpen_amount, blurred, -profile.sharpen_amount, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        midtone_mask = np.clip((gray - 28.0) / 70.0, 0.0, 1.0)
        midtone_mask *= 1.0 - np.clip((gray - 190.0) / 45.0, 0.0, 1.0)
        motion_scale = 1.0 - 0.35 * np.clip(metrics.motion_score / max(self.config.high_motion_threshold, 1.0), 0.0, 1.0)
        blend = (midtone_mask * motion_scale)[:, :, None]

        restored = frame.astype(np.float32) * (1.0 - blend) + sharpened.astype(np.float32) * blend
        return np.clip(restored, 0, 255).astype(np.uint8)

    def _stabilize_bright_luminance(self, frame: np.ndarray, metrics: FrameMetrics) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        if self._previous_luminance is None or self._previous_luminance.shape != l_channel.shape:
            self._previous_luminance = l_channel.copy()
            return frame

        current = l_channel.astype(np.float32)
        previous = self._previous_luminance.astype(np.float32)
        motion = np.clip(metrics.motion_score / max(self.config.high_motion_threshold, 1.0), 0.0, 1.0)
        current_weight = 0.36 + 0.38 * motion
        smoothed = previous * (1.0 - current_weight) + current * current_weight

        bright_mask = np.clip((current - 150.0) / 75.0, 0.0, 1.0)
        stable = current * (1.0 - bright_mask) + smoothed * bright_mask
        stable = np.clip(stable, 0, 255).astype(np.uint8)

        self._previous_luminance = stable.copy()
        return cv2.cvtColor(cv2.merge((stable, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    def _smooth_value(self, key: str, value: float, alpha: float) -> float:
        previous = self._state.get(key)
        if previous is None:
            self._state[key] = value
            return value

        smoothed = previous * (1.0 - alpha) + value * alpha
        self._state[key] = smoothed
        return smoothed
