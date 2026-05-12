import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

class Filters:
    """
    A collection of image filters, designed to be highly modular.
    To add a new filter in the future, simply define a new @staticmethod here 
    that takes a BGR 'frame' as input and returns a modified BGR frame!
    """
    
    # Initialize MediaPipe Face Landmarker (Tasks API)
    _base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    _options = vision.FaceLandmarkerOptions(
        base_options=_base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3  # Lowered to 0.3 so it doesn't give up on far-away faces!
    )
    face_mesh = vision.FaceLandmarker.create_from_options(_options)

    # Dictionary to hold the last known positions for Temporal Smoothing
    _last_points = {}

    @staticmethod
    def smooth_point(name, pt, face_id=0, dynamic_alpha_base=0.3):
        """
        Adaptive Exponential Moving Average (EMA) smoothing.
        If the movement is small (jitter), it smooths heavily.
        If the movement is large (user moved their head), it snaps quickly to avoid lag.
        """
        unique_name = f"{face_id}_{name}"
        if unique_name not in Filters._last_points:
            Filters._last_points[unique_name] = pt
            return pt

        prev_pt = Filters._last_points[unique_name]
        dist = math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1])

        # If movement is larger than 15 pixels, snap faster (alpha closer to 1.0)
        # If movement is small, smooth heavily (alpha closer to 0.1)
        if dist > 15:
            alpha = 0.8
        else:
            alpha = dynamic_alpha_base

        new_x = int(alpha * pt[0] + (1 - alpha) * prev_pt[0])
        new_y = int(alpha * pt[1] + (1 - alpha) * prev_pt[1])
        
        Filters._last_points[unique_name] = (new_x, new_y)
        return new_x, new_y

    
    @staticmethod
    def apply_sepia(frame, face_id=0):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia

    @staticmethod
    def apply_grayscale(frame, face_id=0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_cartoon(frame, face_id=0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    @staticmethod
    def apply_beautify(frame, face_id=0):
        """
        Snapchat-style "Beautify" filter: 
        Smooths skin while preserving edges, and slightly boosts saturation.
        """
        # Bilateral filter smooths flat surfaces (skin) but keeps edges (eyes, lips) sharp
        smoothed = cv2.bilateralFilter(frame, d=15, sigmaColor=75, sigmaSpace=75)
        
        # Convert to HSV to boost saturation slightly for a healthy "glowing" look
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Boost saturation by 20%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def apply_cyberpunk(frame, face_id=0):
        """
        A vibrant Cyberpunk/Synthwave filter with chromatic aberration 
        and heavy magenta/cyan neon color grading.
        """
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Chromatic Aberration: Shift the Red channel slightly to the right, Blue to the left
        shift_amount = 5
        r_shifted = np.roll(r, shift_amount, axis=1)
        b_shifted = np.roll(b, -shift_amount, axis=1)
        
        # Merge back
        shifted_frame = cv2.merge((b_shifted, g, r_shifted))
        
        # Color Grading: Boost Red and Blue (Magenta/Purple vibe)
        shifted_frame = shifted_frame.astype(np.float32)
        shifted_frame[:, :, 0] *= 1.3 # Boost Blue
        shifted_frame[:, :, 1] *= 0.8 # Reduce Green
        shifted_frame[:, :, 2] *= 1.2 # Boost Red
        
        shifted_frame = np.clip(shifted_frame, 0, 255).astype(np.uint8)
        return shifted_frame

    @staticmethod
    def apply_thermal(frame, face_id=0):
        """
        Snapchat thermal camera effect using OpenCV ColorMaps.
        """
        # Convert to grayscale first for a better thermal mapping
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return thermal
        
    @staticmethod
    def apply_pencil_sketch(frame, face_id=0):
        """
        Pencil sketch using Canny edges with soft shading blend.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(smooth, 40, 120)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_inv = cv2.bitwise_not(edges)

        # Shading: classic pencil shading via dodge blend
        blurred = cv2.GaussianBlur(gray, (19, 19), 0)
        inverted = cv2.bitwise_not(blurred)
        shading = cv2.divide(gray, inverted, scale=256)

        # Blend edges into shading for a pencil look
        sketch = cv2.multiply(shading, edges_inv, scale=1 / 255.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_snapchat_prototype(frame, face_id=0):
        """
        Snapchat Filter Prototype with Clown Nose, Googly Eyes, and Joker Smile.
        """
        frame_out = frame.copy()
        h, w, _ = frame_out.shape
        
        # Convert the BGR image to RGB before processing
        rgb_frame = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = Filters.face_mesh.detect(mp_image)

        # Normal solid colors
        solid_white = (255, 255, 255)
        solid_red = (0, 0, 200)
        solid_green = (0, 200, 0)
        solid_black = (0, 0, 0)

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                
                # --- 2. GET EYE DISTANCE FOR PERFECT ADAPTIVE SCALING ---
                left_iris = face_landmarks[468]
                right_iris = face_landmarks[473]
                
                lx, ly = int(left_iris.x * w), int(left_iris.y * h)
                rx, ry = int(right_iris.x * w), int(right_iris.y * h)
                
                lx, ly = Filters.smooth_point('left_iris', (lx, ly), face_id=face_id)
                rx, ry = Filters.smooth_point('right_iris', (rx, ry), face_id=face_id)

                eye_distance = math.hypot(rx - lx, ry - ly)
                if eye_distance <= 0:
                    continue

                # --- 3. THE GOOGLY EYES ---
                googly_radius = int(eye_distance / 2.5)
                pupil_radius = int(googly_radius / 2.5)
                pupil_offset = max(1, int(googly_radius * 0.2))

                # Draw solid eyes on frame_out
                cv2.circle(frame_out, (lx, ly), googly_radius, solid_white, -1, cv2.LINE_AA)
                cv2.circle(frame_out, (lx, ly), googly_radius, solid_black, 2, cv2.LINE_AA)
                cv2.circle(frame_out, (lx, ly - pupil_offset), pupil_radius, solid_black, -1, cv2.LINE_AA)

                cv2.circle(frame_out, (rx, ry), googly_radius, solid_white, -1, cv2.LINE_AA)
                cv2.circle(frame_out, (rx, ry), googly_radius, solid_black, 2, cv2.LINE_AA)
                cv2.circle(frame_out, (rx, ry - pupil_offset), pupil_radius, solid_black, -1, cv2.LINE_AA)

                # --- 4. THE CLOWN NOSE ---
                nose_tip = face_landmarks[4]
                nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
                nx, ny = Filters.smooth_point('nose', (nx, ny), face_id=face_id)
                
                nose_radius = int(eye_distance * 0.45)
                
                # Draw solid nose
                cv2.circle(frame_out, (nx, ny), nose_radius, solid_red, -1, cv2.LINE_AA)
                cv2.circle(frame_out, (nx, ny), nose_radius, solid_black, 2, cv2.LINE_AA)

                # --- 5. THE JOKER SMILE ---
                mouth_left = face_landmarks[61]
                mouth_right = face_landmarks[291]
                mlx, mly = int(mouth_left.x * w), int(mouth_left.y * h)
                mrx, mry = int(mouth_right.x * w), int(mouth_right.y * h)
                
                mlx, mly = Filters.smooth_point('mouth_left', (mlx, mly), face_id=face_id)
                mrx, mry = Filters.smooth_point('mouth_right', (mrx, mry), face_id=face_id)

                smile_dx = int(eye_distance * 0.6)
                smile_dy = int(eye_distance * 0.4)
                line_thickness = max(2, int(eye_distance * 0.08))

                # Draw solid smile
                cv2.line(frame_out, (mlx, mly), (mlx - smile_dx, mly - smile_dy), solid_green, line_thickness, cv2.LINE_AA)
                cv2.line(frame_out, (mrx, mry), (mrx + smile_dx, mry - smile_dy), solid_green, line_thickness, cv2.LINE_AA)

        return frame_out
