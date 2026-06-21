import cv2
from config import IP_WEBCAM_URL, CAMERA_INDEX


def open_camera():
    """Open the configured camera. Returns (cap, True) or (None, False)."""
    source = IP_WEBCAM_URL if IP_WEBCAM_URL else CAMERA_INDEX
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"  [ERROR] Could not open camera (source={source}). "
            "Check CAMERA_INDEX in config.py or use 'Load from image file'."
        )
        return None, False
    return cap, True


def capture_frame_interactive():
    """
    Show a live preview; wait for 'c' (capture) or 'q' (quit).
    Returns a numpy frame or None.
    """
    cap, ok = open_camera()
    if not ok:
        return None

    print("  Press 'c' to capture, 'q' to cancel.")
    captured = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  [WARN] Failed to grab frame. Check camera connection.")
            break

        cv2.imshow("Capture — press C", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            captured = frame.copy()
            break
        elif key == ord("q"):
            print("  Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured