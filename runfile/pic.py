import cv2

VIDEO_PATH = "C:/Users/User/Desktop/rem.mov"   # путь к твоему видео
SAVE_PATH = "frame.jpg"    # куда сохранить

FRAME_NUMBER = 11  # можно менять (0 = первый кадр)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)

ret, frame = cap.read()
if not ret:
    print("Не удалось прочитать кадр.")
else:
    cv2.imwrite(SAVE_PATH, frame)
    print(f"Кадр сохранён как {SAVE_PATH}")
    cv2.imshow("Extracted Frame", frame)
    cv2.waitKey(0)