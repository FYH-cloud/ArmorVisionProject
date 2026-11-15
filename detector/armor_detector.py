import cv2
import numpy as np


class BlueArmorDetector:

    def __init__(self):
        # 蓝色 HSV 阈值
        self.lower_blue = np.array([90, 60, 50])
        self.upper_blue = np.array([140, 255, 255])

        # 灯条亮度阈值
        self.white_lower = np.array([0, 0, 190])
        self.white_upper = np.array([180, 115, 255])

        # 灯条筛选参数
        self.min_area = 40
        self.max_area = 12000
        self.min_length = 20
        self.min_ratio = 2.0

    def preprocess(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 提取蓝色区域
        blue_mask = cv2.inRange(
            hsv,
            self.lower_blue,
            self.upper_blue
        )

        # 对蓝色区域进行膨胀，作为灯条搜索范围
        kernel = np.ones((9, 9), np.uint8)

        blue_roi = cv2.dilate(
            blue_mask,
            kernel,
            iterations=2
        )

        # 提取高亮区域
        white_mask = cv2.inRange(
            hsv,
            self.white_lower,
            self.white_upper
        )

        # 只保留蓝色区域附近的高亮部分
        led_mask = cv2.bitwise_and(
            white_mask,
            blue_roi
        )

        # 形态学处理
        led_mask = cv2.morphologyEx(
            led_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8)
        )

        return led_mask, blue_mask

    def find_light_bars(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        light_bars = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            rect = cv2.minAreaRect(contour)

            (cx, cy), (w, h), angle = rect

            if w == 0 or h == 0:
                continue

            # 统一长边和短边
            length = max(w, h)
            width = min(w, h)

            ratio = length / (width + 1e-6)

            if length < self.min_length:
                continue

            if ratio < self.min_ratio:
                continue

            # 获取灯条四角点
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            light_bars.append({
                "center": (cx, cy),
                "length": length,
                "width": width,
                "angle": angle,
                "box": box,
                "contour": contour
            })

        return light_bars

    def match_light_bars(self, light_bars):

        armors = []

        # 两两匹配
        for i in range(len(light_bars)):

            for j in range(i + 1, len(light_bars)):

                left = light_bars[i]
                right = light_bars[j]

                x1, y1 = left["center"]
                x2, y2 = right["center"]

                # 保证左右顺序
                if x1 > x2:
                    left, right = right, left
                    x1, y1 = left["center"]
                    x2, y2 = right["center"]

                # 两根灯条的高度
                h1 = left["length"]
                h2 = right["length"]

                # 高度相似度
                height_ratio = min(h1, h2) / max(h1, h2)

                if height_ratio < 0.55:
                    continue

                # 中心点垂直偏差
                vertical_diff = abs(y1 - y2)

                if vertical_diff > max(h1, h2) * 0.5:
                    continue

                # 两灯条水平距离
                distance = x2 - x1

                if distance < max(left["width"], right["width"]) * 1.5:
                    continue

                if distance > max(h1, h2) * 5:
                    continue

                # 灯条角点排序
                left_points = self.sort_light_corners(left["box"])
                right_points = self.sort_light_corners(right["box"])

                # 装甲板四个角点：左上、右上、右下、左下
                corners = np.array([
                    left_points[0],
                    right_points[1],
                    right_points[2],
                    left_points[3]
                ], dtype=np.float32)

                armors.append({
                    "corners": corners,
                    "center": (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2
                    )
                })

        return armors

    @staticmethod
    def sort_light_corners(points):

        points = np.array(points, dtype=np.float32)

        # 按 y 坐标分成上下两组
        points = points[np.argsort(points[:, 1])]

        top = points[:2]
        bottom = points[2:]

        # 每组按 x 坐标排序
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        # 左上、右上、右下、左下
        return np.array([
            top[0],
            top[1],
            bottom[1],
            bottom[0]
        ])

    def detect(self, frame):

        output = frame.copy()

        mask, blue_mask = self.preprocess(frame)

        light_bars = self.find_light_bars(mask)

        armors = self.match_light_bars(light_bars)

        # 绘制灯条
        for light in light_bars:

            cv2.drawContours(
                output,
                [light["box"]],
                -1,
                (0, 255, 0),
                2
            )

        # 绘制装甲板
        for armor in armors:

            corners = np.int32(armor["corners"])

            cv2.polylines(
                output,
                [corners],
                True,
                (0, 0, 255),
                3
            )

            center = tuple(
                np.int32(armor["center"])
            )

            cv2.circle(
                output,
                center,
                5,
                (0, 255, 255),
                -1
            )

        cv2.putText(
            output,
            f"Blue contours: {len(light_bars)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Armors: {len(armors)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        return output, armors, mask