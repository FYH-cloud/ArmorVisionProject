import cv2
import numpy as np
import glob
import os

# 棋盘格内角点数量
CHECKERBOARD = (9, 6)

# 每个小格的实际边长，单位mm
SQUARE_SIZE = 26.0

# 创建棋盘格三维坐标
objp = np.zeros(
    (9 * 6, 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:9,
    0:6
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob(
    "calibration/calibration/images/*.jpg"
)

if len(images) == 0:
    raise RuntimeError("没有找到标定照片")

image_size = None

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        continue

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    image_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners2,
            ret
        )

        cv2.imshow(
            "Detected Corners",
            img
        )

        cv2.waitKey(200)

cv2.destroyAllWindows()

if len(objpoints) < 10:
    raise RuntimeError(
        "有效标定图片不足10张，请重新拍摄"
    )

# 执行相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
)

print("相机内参矩阵：")
print(camera_matrix)

print("畸变系数：")
print(dist_coeffs)

print("平均重投影误差：")
print(ret)

# 保存标定参数
os.makedirs("calibration", exist_ok=True)

np.savez(
    "calibration/camera_params.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("标定完成，参数保存成功！")