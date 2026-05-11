import cv2
import numpy as np
from pipeline import VideoPipeline
from filters.registry import FilterRegistry
from filters.image_filters import Filters
from filters.gesture_filters import apply_temperature


def switch_filter(pipeline, filter_name):
    """Helper to switch the active filter by name from the registry."""
    func = FilterRegistry.get(filter_name)
    if func is None:
        print(f"Unknown filter: {filter_name}")
        return
    pipeline.set_filter(
        filter_name,
        func,
        is_face_filter=FilterRegistry.get_is_face_filter(filter_name),
        is_gesture_filter=FilterRegistry.get_is_gesture_filter(filter_name),
    )


def main():
    print("Initializing Unified Video Processing Pipeline...")

    # 1. Instantiate the Pipeline
    pipeline = VideoPipeline()

    # 2. Register all filters
    # --- ImageProject filters ---
    FilterRegistry.register("sepia",          Filters.apply_sepia,               is_face_filter=False)
    FilterRegistry.register("grayscale",      Filters.apply_grayscale,           is_face_filter=False)
    FilterRegistry.register("cartoon",        Filters.apply_cartoon,             is_face_filter=False)
    FilterRegistry.register("beautify",       Filters.apply_beautify,            is_face_filter=True)
    FilterRegistry.register("cyberpunk",      Filters.apply_cyberpunk,           is_face_filter=False)
    FilterRegistry.register("thermal",        Filters.apply_thermal,             is_face_filter=False)
    FilterRegistry.register("pencil_sketch",  Filters.apply_pencil_sketch,       is_face_filter=False)
    FilterRegistry.register("snapchat",       Filters.apply_snapchat_prototype,  is_face_filter=True)
    # --- Gesture-controlled filter ---
    FilterRegistry.register("gesture_temp",   apply_temperature,                 is_face_filter=False, is_gesture_filter=True)

    # Build a list so we can cycle through filters with number keys
    filter_names = FilterRegistry.list_filters()

    # 3. Set default filter
    switch_filter(pipeline, "snapchat")

    # 4. Start Webcam Loop
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n" + "=" * 50)
    print("  UNIFIED FILTER PIPELINE — CONTROLS")
    print("=" * 50)
    print(f"  1-{len(filter_names)}  : Switch filters")
    for i, name in enumerate(filter_names):
        print(f"     [{i + 1}] {name}")
    print("  n / p  : Next / Previous filter")
    print("  ✋ open / fist : Open palm = next, fist = previous")
    print("  m      : Toggle Metrics Overlay")
    print("  h      : Toggle Heatmap")
    print("  g      : Toggle Histogram Graph")
    print("  q      : Quit")
    print("=" * 50 + "\n")

    current_index = filter_names.index("snapchat")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Push frame through our unified pipeline
        final_frame, optimized_frame = pipeline.process_frame(frame)

        # --- Gesture Pose → Filter Switch ---
        if pipeline.gesture_swipe == "right":
            current_index = (current_index + 1) % len(filter_names)
            switch_filter(pipeline, filter_names[current_index])
        elif pipeline.gesture_swipe == "left":
            current_index = (current_index - 1) % len(filter_names)
            switch_filter(pipeline, filter_names[current_index])

        # Build the left panel
        # Always show the hand-tracking annotated frame
        if pipeline.gesture_annotated_frame is not None:
            original_display = pipeline.gesture_annotated_frame.copy()
            cv2.putText(original_display, "Hand Tracking / Gesture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            original_display = frame.copy()
            cv2.putText(original_display, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Match heights for side-by-side
        if final_frame.shape[0] != original_display.shape[0]:
            original_display = cv2.resize(
                original_display,
                (final_frame.shape[1], final_frame.shape[0])
            )

        combined_frame = np.hstack((original_display, final_frame))

        # Draw current filter name on combined view
        cv2.putText(combined_frame, f"Filter: {pipeline.active_filter_name}",
                    (combined_frame.shape[1] // 2 + 10, combined_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Unified Filter Pipeline", combined_frame)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            pipeline.show_metrics = not pipeline.show_metrics
        elif key == ord('h'):
            pipeline.show_heatmap = not pipeline.show_heatmap
        elif key == ord('g'):
            pipeline.show_histogram = not pipeline.show_histogram
        elif key == ord('n'):
            # Next filter
            current_index = (current_index + 1) % len(filter_names)
            switch_filter(pipeline, filter_names[current_index])
        elif key == ord('p'):
            # Previous filter
            current_index = (current_index - 1) % len(filter_names)
            switch_filter(pipeline, filter_names[current_index])
        elif ord('1') <= key <= ord('9'):
            # Number key to jump to a specific filter
            idx = key - ord('1')
            if idx < len(filter_names):
                current_index = idx
                switch_filter(pipeline, filter_names[current_index])

    # Clean up resources
    pipeline.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
