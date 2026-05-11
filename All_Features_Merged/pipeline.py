import time
import cv2
import numpy as np
from core.face_detector import FaceDetector
from core.enhancer import FaceEnhancer
from core.gesture_detector import GestureDetector
from core.intensity.analyzer import FrameQualityAnalyzer, FrameMetrics
from core.intensity.config import EnhancementConfig
from core.intensity.enhancer import AdaptiveFrameEnhancer
from core.intensity.utils import draw_overlay, compose_with_heatmap, compose_with_histogram
from filters.gesture_filters import draw_thermometer

class VideoPipeline:
    """
    The central orchestration class for the Unified Video Enhancement project.
    This coordinates face detection, gesture recognition, AI enhancement, and filter application.
    """
    def __init__(self):
        # 1. Initialize modules
        self.face_detector = FaceDetector(model_path="blaze_face_full_range.tflite", min_detection_confidence=0.4)
        self.enhancer = FaceEnhancer(model_path="FSRCNN_x4.pb", scale=4)
        self.gesture_detector = GestureDetector(model_path="hand_landmarker.task")
        
        # Gesture state
        self.gesture_temperature = 0.0
        self.gesture_swipe = None          # "left", "right", or None
        self.gesture_annotated_frame = None
        
        # 2. Intensity Optimizer
        self.intensity_config = EnhancementConfig()
        self.intensity_analyzer = FrameQualityAnalyzer()
        self.intensity_enhancer = AdaptiveFrameEnhancer(self.intensity_config)
        self._last_processing_ms = 0.0
        self._mode = "balanced"
        self._overload_frames = 0
        self._low_light_frames = 0
        
        # 3. Pipeline UI config
        self.show_metrics = True
        self.show_heatmap = False
        self.show_histogram = False
        
        # 4. Pipeline State
        self.active_filter_name = None
        self.active_filter_func = None
        self.is_face_filter = True
        self.is_gesture_filter = False
        self.area_threshold = 0.05
        
        # Dictionary to store persistent data
        self.state = {
            "last_points": {},
            "face_ids": []
        }

    def set_filter(self, filter_name, filter_func, is_face_filter=True, is_gesture_filter=False):
        """Change the currently active filter."""
        print(f"Switching to filter: {filter_name} (Face: {is_face_filter}, Gesture: {is_gesture_filter})")
        self.active_filter_name = filter_name
        self.active_filter_func = filter_func
        self.is_face_filter = is_face_filter
        self.is_gesture_filter = is_gesture_filter

    def _select_mode(self, metrics):
        overloaded = (
            metrics.processing_ms > self.intensity_config.frame_budget_ms * 2.4
            or metrics.motion_score > self.intensity_config.high_motion_threshold * 2.2
        )
        low_light = (
            metrics.shadow_level < self.intensity_config.shadow_level_threshold
            or metrics.shadow_ratio > self.intensity_config.shadow_ratio_threshold
            or metrics.brightness < self.intensity_config.low_light_threshold
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

    def get_face_mask(self, h, w, orig_x, orig_y, orig_bw, orig_bh):
        """Generates an elliptical mask that covers the entire face."""
        mask = np.zeros((h, w), dtype=np.float32)
        cx = orig_x + orig_bw // 2
        cy = orig_y + orig_bh // 2
        axes = (int(orig_bw * 0.6), int(orig_bh * 0.75))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
        k_size = int(min(orig_bw, orig_bh) * 0.5) | 1
        mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
        return np.expand_dims(mask, axis=-1)

    def process_frame(self, frame):
        """
        Main processing loop for a single frame.
        """
        if not self.active_filter_func:
            return frame, frame
            
        start_time = time.perf_counter()
        
        # --- PHASE 0: Intensity Optimization ---
        # Optimizes low-light, contrast, and noise.
        # This makes the face detector and FSRCNN work much better in dark rooms!
        metrics = self.intensity_analyzer.analyze(frame, self._last_processing_ms)
        self._mode = self._select_mode(metrics)
        optimized_frame = self.intensity_enhancer.enhance(frame, metrics, self._mode)
            
        h, w = optimized_frame.shape[:2]
        frame_area = h * w
        
        final_frame = optimized_frame.copy()
        
        # --- PHASE 1a: Gesture Detection ---
        # Always run so swipe events are detected regardless of active filter
        self.gesture_temperature, self.gesture_swipe, self.gesture_annotated_frame = \
            self.gesture_detector.process(optimized_frame)
        
        # --- PHASE 1b: Face Detection ---
        # Only run detection if the filter requires it
        bboxes, detections = [], []
        if self.is_face_filter:
            _, detections, bboxes = self.face_detector.process(optimized_frame.copy(), draw=False)

        # --- PHASE 2: Apply Filters & Enhancement ---
        if self.is_face_filter and bboxes:
            # Sort left to right to maintain consistent IDs across frames
            zipped = list(zip(bboxes, detections))
            zipped.sort(key=lambda x: x[0][0])

            for face_id, (bbox, detection) in enumerate(zipped):
                orig_x, orig_y, orig_bw, orig_bh = bbox

                # Expand the bounding box
                cx = orig_x + orig_bw // 2
                cy = orig_y + orig_bh // 2
                new_bw = int(orig_bw * 1.3)
                new_bh = int(orig_bh * 1.5)
                new_cy = cy - int(orig_bh * 0.1)
                
                x = max(0, cx - new_bw // 2)
                y = max(0, new_cy - new_bh // 2)
                bw = min(new_bw, w - x)
                bh = min(new_bh, h - y)

                if bw <= 0 or bh <= 0:
                    continue

                original_area_ratio = (orig_bw * orig_bh) / frame_area
                roi = optimized_frame[y:y+bh, x:x+bw]
                
                # FSRCNN Logic based on threshold
                needs_enhancement = (original_area_ratio < self.area_threshold)
                
                if needs_enhancement:
                    processed_roi = self.enhancer.enhance(roi)
                    status_text = "FSRCNN+Filter"
                    color = (0, 255, 0)
                else:
                    processed_roi = roi.copy()
                    status_text = "Raw+Filter"
                    color = (0, 255, 255)
                    
                # Apply filter strictly to the local face ROI
                filtered_roi = self.active_filter_func(processed_roi, face_id=face_id)
                
                # Resize back to original ROI size
                if filtered_roi.shape[:2] != (bh, bw):
                    filtered_roi = cv2.resize(filtered_roi, (bw, bh), interpolation=cv2.INTER_AREA)
                
                # Alpha Blending
                full_mask = self.get_face_mask(h, w, orig_x, orig_y, orig_bw, orig_bh)
                roi_mask = full_mask[y:y+bh, x:x+bw]
                
                original_roi_float = final_frame[y:y+bh, x:x+bw].astype(np.float32)
                filtered_roi_float = filtered_roi.astype(np.float32)
                
                blended_roi = (filtered_roi_float * roi_mask + original_roi_float * (1.0 - roi_mask)).astype(np.uint8)
                final_frame[y:y+bh, x:x+bw] = blended_roi
                    
                cv2.putText(final_frame, status_text, (x, max(20, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
        elif self.is_gesture_filter:
            # Gesture-controlled filter: pass temperature to the filter function
            final_frame = self.active_filter_func(final_frame, self.gesture_temperature)
            
            # Draw the thermometer UI on top
            draw_thermometer(final_frame, self.gesture_temperature)
            cv2.putText(final_frame, "Gesture Temperature", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
        elif not self.is_face_filter:
            # Full-frame filter (no face detection, no FSRCNN needed)
            final_frame = self.active_filter_func(final_frame, face_id=0)
                        
        self._last_processing_ms = (time.perf_counter() - start_time) * 1000.0
        
        full_metrics = FrameMetrics(
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
        
        # --- PHASE 5: UI Overlays ---
        if self.show_metrics:
            draw_overlay(final_frame, full_metrics, self._mode)
            
        if self.show_heatmap:
            if final_frame.shape[:2] != frame.shape[:2]:
                display_frame = optimized_frame.copy()
            else:
                display_frame = final_frame
            final_frame = compose_with_heatmap(display_frame, frame, optimized_frame, display_frame.shape[0])
            
        if self.show_histogram:
            if final_frame.shape[:2] != frame.shape[:2]:
                display_frame = optimized_frame.copy()
            else:
                display_frame = final_frame
            final_frame = compose_with_histogram(display_frame, frame, optimized_frame, int(display_frame.shape[0] * 0.25))
            
        return final_frame, optimized_frame

    def release(self):
        """Free any loaded resources, models, or video captures."""
        if hasattr(self.face_detector, 'close') and self.face_detector:
            self.face_detector.close()
        if hasattr(self.gesture_detector, 'close') and self.gesture_detector:
            self.gesture_detector.close()
