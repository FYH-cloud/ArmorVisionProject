import numpy as np
import cv2
from pathlib import Path

# 获取参数文件路径
BASE_DIR = Path(__file__).resolve().parent
PARAM_PATH = BASE_DIR / "calibration/camera_params.npz"

# 检查文件是否存在
if not PARAM_PATH.exists():
    raise FileNotFoundError(
        f"找不到相机参数文件：{PARAM_PATH}"
    )

# 读取相机参数
data = np.load(str(PARAM_PATH))

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("========== 相机参数读取成功 ==========")

print("相机内参矩阵 K：")
print(camera_matrix)

print("\n畸变系数 D：")
print(dist_coeffs)

print("\n参数文件路径：")
print(PARAM_PATH)

print("\n相机标定参数验证完成！")