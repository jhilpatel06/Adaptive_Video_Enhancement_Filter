import cv2
import time
import numpy as np
from cv2 import dnn_superres


# Initialize the CLAHE object outside your webcam loop so it doesn't recreate it every frame
# clipLimit: The threshold for contrast limiting (usually 2.0 to 4.0 is a good sweet spot).
# tileGridSize: Divides the image into an 8x8 grid to calculate contrast locally.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])



def calculate_psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    print("Initializing FSRCNN Model...")
    # Initialize Super Resolution
    sr = dnn_superres.DnnSuperResImpl_create()
    model_path = "FSRCNN_x3.pb"
    scale = 3
    
    try:
        sr.readModel(model_path)
        sr.setModel("fsrcnn", scale)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure FSRCNN_x3.pb is in the current directory.")
        return

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting video feed. Press 'q' to quit.")

    fps = 0
    fps_smoothing = 0.9

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # --- PREPARE IMAGES ---
        # We treat the webcam frame as the "Ground Truth" High-Resolution image.
        # To simulate a low-resolution input, we downscale it.
        # FSRCNN can be computationally heavy, so we crop the center of the frame 
        # to ensure real-time performance on standard CPUs.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # 2. Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 3. Apply CLAHE strictly to the L (Lightness) channel
        cl = clahe.apply(l_channel)

        # 4. Merge the CLAHE-enhanced L channel back with the original A and B channels
        merged_lab = cv2.merge((cl, a_channel, b_channel))

        # 5. Convert the image back to BGR so it's ready for FSRCNN and MediaPipe
        frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


        h, w = frame.shape[:2]
        target_h, target_w = 480, 480 # Adjust these for different performance/quality trade-offs
        
        # Crop center
        start_y = max(0, h // 2 - target_h // 2)
        start_x = max(0, w // 2 - target_w // 2)
        gt_frame = frame[start_y:start_y+target_h, start_x:start_x+target_w]

        # Ensure dimensions are exactly divisible by the scale factor
        gt_h = (gt_frame.shape[0] // scale) * scale
        gt_w = (gt_frame.shape[1] // scale) * scale
        gt_frame = gt_frame[:gt_h, :gt_w]

        # 1. Downscale to create the Low Resolution (LR) input image
        lr_h, lr_w = gt_h // scale, gt_w // scale
        lr_frame = cv2.resize(gt_frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        # --- UPSCALING ---
        # 2. Upscale using standard Bicubic interpolation (Baseline)
        bicubic_frame = cv2.resize(lr_frame, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
        # de_gridded_frame = cv2.medianBlur(bicubic_frame, 3)
        # blurred = cv2.GaussianBlur(de_gridded_frame, (0, 0), 3)
        # bicubic_frame = cv2.addWeighted(de_gridded_frame, 1.5, blurred, -0.5, 0)

        # 3. Upscale using FSRCNN Neural Network
        fsrcnn_frame = sr.upsample(lr_frame)
        # de_gridded_frame = cv2.medianBlur(fsrcnn_frame, 3)
        # blurred = cv2.GaussianBlur(de_gridded_frame, (0, 0), 3)
        # fsrcnn_frame = cv2.addWeighted(de_gridded_frame, 1.5, blurred, -0.5, 0)

        # --- METRICS ---
        # 4. Calculate quality metrics (PSNR)
        psnr_bicubic = calculate_psnr(gt_frame, bicubic_frame)
        psnr_fsrcnn = calculate_psnr(gt_frame, fsrcnn_frame)

        # --- DISPLAY ---
        # 5. Add text overlays for metrics and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        # Background rectangle for text readability
        def draw_text_with_bg(img, text, pos):
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img, (pos[0]-2, pos[1]-th-2), (pos[0]+tw+2, pos[1]+2), (0,0,0), -1)
            cv2.putText(img, text, pos, font, font_scale, color, thickness)

        gt_display = gt_frame.copy()
        draw_text_with_bg(gt_display, "Ground Truth", (10, 20))
        draw_text_with_bg(gt_display, f"FPS: {fps:.1f}", (10, 40))

        bicubic_display = bicubic_frame.copy()
        draw_text_with_bg(bicubic_display, "Bicubic Upscale", (10, 20))
        draw_text_with_bg(bicubic_display, f"PSNR: {psnr_bicubic:.2f} dB", (10, 40))

        fsrcnn_display = fsrcnn_frame.copy()
        draw_text_with_bg(fsrcnn_display, "FSRCNN Upscale", (10, 20))
        draw_text_with_bg(fsrcnn_display, f"PSNR: {psnr_fsrcnn:.2f} dB", (10, 40))

        # Concatenate horizontally to display side-by-side
        combined_frame = np.hstack((gt_display, bicubic_display, fsrcnn_display))
        cv2.imshow("FSRCNN Real-Time Enhancement", combined_frame)

        # Update FPS
        end_time = time.time()
        processing_time = end_time - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        # Exponential moving average for smoother FPS display
        fps = (fps * fps_smoothing) + (current_fps * (1.0 - fps_smoothing))

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
