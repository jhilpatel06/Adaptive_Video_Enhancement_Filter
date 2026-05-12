import cv2
import numpy as np
from enhancer import FaceEnhancer


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two BGR images.
    Both must be the same shape. Returns float('inf') if identical.
    """
    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)
    mse = np.mean((img_a - img_b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def main():
    print("Initializing FSRCNN Downsample Demo...")
    enhancer = FaceEnhancer(model_path="FSRCNN_x4.pb", scale=4)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Input to FSRCNN: 160x120  ->  FSRCNN x4 output: 640x480
    down_w, down_h = 160, 120

    PANEL_W, PANEL_H = 640, 480  # display size per panel

    def make_panel(img, title, psnr_val=None, interp=cv2.INTER_LINEAR):
        """Resize to panel size, add a label bar with title + optional PSNR."""
        panel = cv2.resize(img, (PANEL_W, PANEL_H), interpolation=interp)
        bar_h = 44
        cv2.rectangle(panel, (0, PANEL_H - bar_h), (PANEL_W, PANEL_H), (0, 0, 0), -1)
        if psnr_val is not None:
            psnr_str = "inf" if psnr_val == float("inf") else f"{psnr_val:.2f} dB"
            label = f"{title}   PSNR: {psnr_str}"
        else:
            label = title
        cv2.putText(panel, label, (10, PANEL_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Upscale resolution: x4 model gives 640x480 from 160x120
        up_w, up_h = down_w * 4, down_h * 4

        # Ground truth: center-crop the original to match the upscale size
        frame_h, frame_w = frame.shape[:2]
        if frame_w < up_w or frame_h < up_h:
            frame = cv2.resize(frame, (max(frame_w, up_w), max(frame_h, up_h)), interpolation=cv2.INTER_LINEAR)
            frame_h, frame_w = frame.shape[:2]

        start_x = (frame_w - up_w) // 2
        start_y = (frame_h - up_h) // 2
        reference = frame[start_y:start_y + up_h, start_x:start_x + up_w]

        # Downsample using INTER_AREA to generate the low-res input
        low_res = cv2.resize(reference, (down_w, down_h), interpolation=cv2.INTER_AREA)

        # Nearest-neighbour upscale (blocky, shows raw downsampled quality)
        nearest_up = cv2.resize(low_res, (up_w, up_h), interpolation=cv2.INTER_NEAREST)

        # Bicubic upscale
        bicubic_up = cv2.resize(low_res, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

        # FSRCNN upscale
        fsrcnn_up = enhancer.enhance(low_res)
        # Safety guard: ensure FSRCNN output matches expected size
        if fsrcnn_up.shape[:2] != (up_h, up_w):
            fsrcnn_up = cv2.resize(fsrcnn_up, (up_w, up_h), interpolation=cv2.INTER_LINEAR)

        # PSNR computed at actual upscale resolution (not display size) for accuracy
        psnr_down    = compute_psnr(reference, nearest_up)
        psnr_bicubic = compute_psnr(reference, bicubic_up)
        psnr_fsrcnn  = compute_psnr(reference, fsrcnn_up)

        # Build panels — two rows: Downsampled | Bicubic, then FSRCNN centered
        p_down    = make_panel(nearest_up, "Downsampled", psnr_down,    cv2.INTER_NEAREST)
        p_bicubic = make_panel(bicubic_up, "Bicubic x4",  psnr_bicubic, cv2.INTER_LINEAR)
        p_fsrcnn  = make_panel(fsrcnn_up,  "FSRCNN x4",   psnr_fsrcnn,  cv2.INTER_LINEAR)

        top_row = np.hstack((p_down, p_bicubic))
        side_pad = np.zeros((PANEL_H, PANEL_W // 2, 3), dtype=np.uint8)
        bottom_row = np.hstack((side_pad, p_fsrcnn, side_pad))
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow("FSRCNN Downsample Demo", grid)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
