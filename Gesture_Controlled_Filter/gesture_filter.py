import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

# Initialize mediapipe hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       min_hand_detection_confidence=0.7,
                                       min_hand_presence_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

def apply_temperature_filter(image, temperature):
    """
    Apply a cool/warm filter based on temperature.
    Temperature range is from -100 (Coolest) to 100 (Warmest).
    """
    # Create a copy of the image to modify
    filtered = image.copy().astype(np.float32)
    
    if temperature > 0:
        # Warm filter: Increase Red, Decrease Blue
        r_change = (temperature / 100.0) * 60
        b_change = (temperature / 100.0) * -60
        filtered[:, :, 2] += r_change  # Red channel
        filtered[:, :, 0] += b_change  # Blue channel
    elif temperature < 0:
        # Cool filter: Increase Blue, Decrease Red
        r_change = (abs(temperature) / 100.0) * -60
        b_change = (abs(temperature) / 100.0) * 60
        filtered[:, :, 2] += r_change
        filtered[:, :, 0] += b_change
        
    # Clip values to be between 0 and 255
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

def main():
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    # Range for distance between thumb and index finger
    MIN_DIST = 20
    MAX_DIST = 200
    
    print("Webcam opened. Press 'q' to quit.")
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
            
        # Flip the image horizontally for a selfie-view display
        img = cv2.flip(img, 1)
        original_img = img.copy()
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)
        
        temperature = 0
        dist = 0
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Note: Drawing all landmarks is omitted in the new Tasks API 
                # without extra helper code, but we still draw the required 
                # line between thumb and index finger below.
                
                # Get coordinates for thumb tip (4) and index finger tip (8)
                h, w, c = img.shape
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw circles on the tips and a line between them
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Calculate distance
                dist = math.hypot(x2 - x1, y2 - y1)
                
                # Map distance to temperature (-100 to 100)
                dist_clamped = max(MIN_DIST, min(MAX_DIST, dist))
                temperature = np.interp(dist_clamped, [MIN_DIST, MAX_DIST], [-100, 100])
                
                # Show distance value at the center of the line
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(img, f'Dist: {int(dist)}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Apply filter to the original image based on calculated temperature
        filtered_img = apply_temperature_filter(original_img, temperature)
        
        # Draw the thermometer scale on the filtered image
        bar_x, bar_y = 50, 100
        bar_w, bar_h = 30, 200
        
        # Draw background bar (gray)
        cv2.rectangle(filtered_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)
        
        # Draw colored part based on temperature
        fill_y = int(np.interp(temperature, [-100, 100], [bar_y + bar_h, bar_y]))
        
        if temperature > 0:
            bar_color = (0, 0, 255) # Red for Warm
        elif temperature < 0:
            bar_color = (255, 0, 0) # Blue for Cool
        else:
            bar_color = (0, 255, 0) # Green for Neutral
            
        cv2.rectangle(filtered_img, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(filtered_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), 2)
        
        # Add labels
        cv2.putText(filtered_img, "Warm", (bar_x - 15, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(filtered_img, "Cool", (bar_x - 10, bar_y + bar_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(filtered_img, f"{int(temperature)}", (bar_x + 40, fill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
        
        # Combine both images side by side
        combined = np.hstack((img, filtered_img))
        
        # Add titles
        w_half = img.shape[1]
        cv2.putText(combined, "Original + Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Filtered Video", (w_half + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Gesture Controlled Cool/Warm Filter", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
