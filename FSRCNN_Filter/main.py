import cv2
import numpy as np
from face_detector import FaceDetector
from enhancer import FaceEnhancer
from filters import Filters

class FilterPipeline:
    def __init__(self, filter_func=Filters.apply_sepia):
        """
        Initializes the pipeline with required modules.
        
        Args:
            filter_func: The filter function from Filters class to apply.
        """
        self.detector = FaceDetector(model_path="blaze_face_full_range.tflite", min_detection_confidence=0.4)
        self.enhancer = FaceEnhancer(model_path="FSRCNN_x4.pb", scale=4)
        self.current_filter = filter_func
        self.area_threshold = 0.05  # 5% threshold for face area vs frame area
        
    def get_face_mask(self, frame, bbox):
        """
        Generates an elliptical mask that covers the entire face (forehead to chin).
        This provides a full-face mask without the harsh corners of a box.
        """
        h, w = frame.shape[:2]
        orig_x, orig_y, orig_bw, orig_bh = bbox
        
        # Create black mask
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Center of the face bounding box
        cx = orig_x + orig_bw // 2
        cy = orig_y + orig_bh // 2
        
        # Ellipse axes (width and height of the face)
        axes = (int(orig_bw * 0.6), int(orig_bh * 0.75))  # larger to cover forehead/chin
        
        # Draw solid white ellipse
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
        
        # Soften edges heavily so blending looks natural
        k_size = int(min(orig_bw, orig_bh) * 0.3) | 1
        mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
        
        # Expand dims for broadcasting (h, w, 1)
        mask = np.expand_dims(mask, axis=-1)
        
        return mask

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_area = h * w

        # 1. MediaPipe Face Detection
        _, detections, bboxes = self.detector.process(frame.copy(), draw=False)

        if not bboxes:
            # If no face is detected, just return the original frame
            return frame

        # Sort faces left to right to maintain consistent IDs across frames
        zipped = list(zip(bboxes, detections))
        zipped.sort(key=lambda x: x[0][0])
        
        final_frame = frame.copy()

        for face_id, (bbox, detection) in enumerate(zipped):
            orig_x, orig_y, orig_bw, orig_bh = bbox

            # Expand the bounding box
            cx = orig_x + orig_bw // 2
            cy = orig_y + orig_bh // 2
            
            new_bw = int(orig_bw * 1.3)
            new_bh = int(orig_bh * 1.5)
            new_cy = cy - int(orig_bh * 0.1)  # Shift up to get more forehead
            
            x = max(0, cx - new_bw // 2)
            y = max(0, new_cy - new_bh // 2)
            
            bw = min(new_bw, w - x)
            bh = min(new_bh, h - y)

            if bw <= 0 or bh <= 0:
                continue

            original_area_ratio = (orig_bw * orig_bh) / frame_area

            # Crop face ROI
            roi = frame[y:y+bh, x:x+bw]
            
            # 2. Decide whether to use FSRCNN Enhancement
            needs_enhancement = (original_area_ratio < self.area_threshold)
            
            if needs_enhancement:
                processed_roi = self.enhancer.enhance(roi)
                status_text = f"FSRCNN+Filter"
                color = (0, 255, 0)
            else:
                processed_roi = roi.copy()
                status_text = f"Raw+Filter"
                color = (0, 255, 255)
                
            # 3. Apply filter strictly to the local face ROI
            # Pass face_id so stateful filters can track faces independently
            filtered_roi = self.current_filter(processed_roi, face_id=face_id)
            
            # Resize back to original ROI size to allow seamless clone back into the frame
            if filtered_roi.shape[:2] != (bh, bw):
                filtered_roi = cv2.resize(filtered_roi, (bw, bh), interpolation=cv2.INTER_AREA)
            
            # 4. Use the full-face elliptical mask for Alpha Blending
            full_mask = self.get_face_mask(frame, (orig_x, orig_y, orig_bw, orig_bh))
            roi_mask = full_mask[y:y+bh, x:x+bw]
            
            # Convert to float for precise blending calculations
            original_roi_float = final_frame[y:y+bh, x:x+bw].astype(np.float32)
            filtered_roi_float = filtered_roi.astype(np.float32)
            
            # Blend: (Filtered * mask) + (Original * (1 - mask))
            blended_roi = (filtered_roi_float * roi_mask + original_roi_float * (1.0 - roi_mask)).astype(np.uint8)
            
            # Put the perfectly feathered face back into the frame
            final_frame[y:y+bh, x:x+bw] = blended_roi
                
            # Optional debug overlay
            cv2.putText(final_frame, status_text, (x, max(20, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return final_frame

    def release(self):
        """Free resources."""
        self.detector.close()

def main():
    print("Initializing Smart Filter Pipeline...")
    
    # You can easily swap filters here by passing a different static method
    # Options: Filters.apply_beautify, Filters.apply_cyberpunk, Filters.apply_thermal, Filters.apply_snapchat_prototype, etc.
    pipeline = FilterPipeline(filter_func=Filters.apply_snapchat_prototype)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Pipeline started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame through the defined flow
        final_frame = pipeline.process_frame(frame)
        
        # Add a label to the original frame
        original_display = frame.copy()
        cv2.putText(original_display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create side-by-side comparison
        combined_frame = np.hstack((original_display, final_frame))
        
        cv2.imshow("Original vs Smart Filter Pipeline", combined_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
