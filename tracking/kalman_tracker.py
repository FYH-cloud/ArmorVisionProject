import cv2
import numpy as np


class KalmanTracker:

    def __init__(self):

        # 6维状态，3维测量
        # 状态：[x, y, z, vx, vy, vz]
        # 测量：[x, y, z]
        self.kf = cv2.KalmanFilter(6, 3)

        # 状态转移矩阵
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)

        # 测量矩阵：只测量位置
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # 过程噪声
        self.kf.processNoiseCov = (
            np.eye(6, dtype=np.float32) * 1e-2
        )

        # 测量噪声
        self.kf.measurementNoiseCov = (
            np.eye(3, dtype=np.float32) * 1e-1
        )

        # 初始误差协方差
        self.kf.errorCovPost = (
            np.eye(6, dtype=np.float32)
        )

        self.initialized = False

    def update(self, position, dt):

        """
        position: [x, y, z]
        dt: 两次测量之间的时间间隔，单位秒
        """

        x, y, z = position

        # 防止时间间隔异常
        dt = max(0.001, min(float(dt), 0.2))

        # 更新状态转移矩阵
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        measurement = np.array([
            [x],
            [y],
            [z]
        ], dtype=np.float32)

        # 第一次测量：初始化位置，速度置零
        if not self.initialized:

            self.kf.statePost = np.array([
                [x],
                [y],
                [z],
                [0],
                [0],
                [0]
            ], dtype=np.float32)

            self.initialized = True

            return self.get_state()

        # 预测
        self.kf.predict()

        # 使用当前测量修正预测
        self.kf.correct(measurement)

        return self.get_state()

    def get_state(self):

        state = self.kf.statePost

        return {
            "x": float(state[0, 0]),
            "y": float(state[1, 0]),
            "z": float(state[2, 0]),
            "vx": float(state[3, 0]),
            "vy": float(state[4, 0]),
            "vz": float(state[5, 0])
        }

    def predict_future(self, seconds=0.1):

        """
        预测未来 seconds 秒的位置
        """

        if not self.initialized:
            return None

        state = self.kf.statePost

        x = float(state[0, 0])
        y = float(state[1, 0])
        z = float(state[2, 0])

        vx = float(state[3, 0])
        vy = float(state[4, 0])
        vz = float(state[5, 0])

        t = float(seconds)

        predicted_position = np.array([
            x + vx * t,
            y + vy * t,
            z + vz * t
        ], dtype=np.float32)

        return predicted_position