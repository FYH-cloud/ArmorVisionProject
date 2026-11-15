import cv2
import numpy as np


class PNPSolver:

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        armor_width,
        armor_height
    ):

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.W = armor_width
        self.H = armor_height

        # 装甲板四个角点的三维坐标
        # 顺序：左上、右上、右下、左下
        self.object_points = np.array([
            [-self.W / 2, -self.H / 2, 0],
            [self.W / 2, -self.H / 2, 0],
            [self.W / 2, self.H / 2, 0],
            [-self.W / 2, self.H / 2, 0]
        ], dtype=np.float32)

    def solve(self, image_points):

        image_points = np.asarray(
            image_points,
            dtype=np.float32
        ).reshape(4, 2)

        # PnP位姿解算
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # 旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 提取三维坐标
        x = float(tvec[0][0])
        y = float(tvec[1][0])
        z = float(tvec[2][0])

        # 计算直线距离
        distance = float(np.linalg.norm(tvec))

        return {
            "rvec": rvec,
            "tvec": tvec,
            "rotation_matrix": rotation_matrix,
            "x": x,
            "y": y,
            "z": z,
            "distance": distance
        }