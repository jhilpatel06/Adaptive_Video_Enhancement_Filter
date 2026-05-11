import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    def __init__(self, model_path='blaze_face_short_range.tflite', min_detection_confidence=0.5):
        """
        Initialize the MediaPipe Face Detector using the modern Tasks API.
        
        Args:
            model_path: Path to the downloaded .tflite model file.
            min_detection_confidence: Minimum confidence value for face detection.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options,
                                             min_detection_confidence=min_detection_confidence)
        self.detector = vision.FaceDetector.create_from_options(options)

    def process(self, image, draw=True):
        """
        Detect faces in the given image.
        
        Args:
            image: The BGR image (OpenCV format) to process.
            draw: Whether to draw bounding boxes on the image.
            
        Returns:
            image: The processed image (with drawings if draw=True).
            detections: A list of detection objects.
            bboxes: A list of bounding boxes in pixel coordinates (x, y, w, h).
        """
        # MediaPipe Tasks API expects RGB images encapsulated in an mp.Image object
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Process the image and detect faces
        detection_result = self.detector.detect(mp_image)
        
        detections = []
        bboxes = []
        
        if detection_result.detections:
            for detection in detection_result.detections:
                detections.append(detection)
                
                # Extract bounding box (it's in absolute pixel coordinates)
                bbox_obj = detection.bounding_box
                x = int(bbox_obj.origin_x)
                y = int(bbox_obj.origin_y)
                w = int(bbox_obj.width)
                h = int(bbox_obj.height)
                bboxes.append((x, y, w, h))
                
                if draw:
                    # Draw bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
        return image, detections, bboxes

    def close(self):
        """
        Free up resources used by the face detection model.
        """
        self.detector.close()

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # Ensure blaze_face_short_range.tflite is in the same directory
    detector = FaceDetector(min_detection_confidence=0.6)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Use the modular face detector
        processed_frame, _, bboxes = detector.process(frame, draw=True)
            
        cv2.imshow('MediaPipe Face Detection', processed_frame)
        if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
            break
            
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
