import cv2

# 打开默认摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头打开失败！")
    exit()

print("摄像头启动成功！")

while True:
    ret, frame = cap.read()

    if not ret:
        print("读取图像失败！")
        break

    cv2.imshow("Camera Test", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()