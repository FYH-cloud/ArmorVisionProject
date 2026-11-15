import cv2
from pathlib import Path

# 创建保存图片的文件夹
save_dir = Path("detector/images")
save_dir.mkdir(parents=True, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("摄像头打开失败")

print("摄像头启动成功")
print("按 S 保存装甲板照片")
print("按 Q 退出")

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("读取画面失败")
        break

    cv2.imshow("Armor Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        count += 1
        filename = save_dir / f"armor_{count:02d}.jpg"
        cv2.imwrite(str(filename), frame)
        print("图片已保存：", filename)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()