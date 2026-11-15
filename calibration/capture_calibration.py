import cv2
import os

save_dir = "calibration/images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头打开失败")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    cv2.putText(
        display,
        f"Saved: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "Press SPACE to save, Q to quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Calibration Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = os.path.join(
            save_dir,
            f"calib_{count:02d}.jpg"
        )

        cv2.imwrite(filename, frame)
        count += 1
        print("保存成功:", filename)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("总共采集照片:", count)