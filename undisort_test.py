import cv2
import numpy as np
from pathlib import Path

# 1. 获取项目路径
BASE_DIR = Path(__file__).resolve().parent
PARAM_PATH = BASE_DIR / "calibration/camera_params.npz"

# 2. 加载相机参数
data = np.load(str(PARAM_PATH))

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("相机参数加载成功")

# 3. 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("摄像头打开失败")

# 与标定时保持相同分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("摄像头启动成功，按 Q 退出")

while True:
    ret, frame = cap.read()

    if not ret:
        print("摄像头读取失败")
        break

    # 4. 获取最佳校正相机矩阵
    h, w = frame.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    # 5. 去畸变
    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    # 6. 显示原始画面和校正画面
    comparison = np.hstack((frame, undistorted))

    cv2.imshow("Original | Undistorted", comparison)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("去畸变测试结束")