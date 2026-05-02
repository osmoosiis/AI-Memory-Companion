import cv2 as cv

for i in range(3):
    cap = cv.VideoCapture(i)

    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
    else:
        print(f"No camera at index {i}")