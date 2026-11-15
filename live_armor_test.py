import cv2
import numpy as np
import time
from pathlib import Path

# 导入装甲板检测模块
from detector.armor_detector import BlueArmorDetector

# 导入PnP位姿解算模块
from pose.pnp_solver import PNPSolver

# 导入卡尔曼滤波模块
from tracking.kalman_tracker import KalmanTracker


# ==================================================
# 1. 项目路径
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

possible_paths = [
    BASE_DIR / "calibration" / "camera_params.npz",
    BASE_DIR / "camera_params.npz"
]

PARAM_PATH = None

for path in possible_paths:
    if path.exists():
        PARAM_PATH = path
        break

if PARAM_PATH is None:
    raise FileNotFoundError(
        "找不到 camera_params.npz，请检查文件位置！"
    )


# ==================================================
# 2. 读取相机标定参数
# ==================================================

data = np.load(str(PARAM_PATH))

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("相机参数读取成功！")
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)


# ==================================================
# 3. 设置装甲板实际尺寸
# ==================================================

# 单位：mm
# 注意：这两个值目前是临时演示值
# 正式测试时必须替换成检测四角点对应的真实尺寸

ARMOR_WIDTH = 160.0
ARMOR_HEIGHT = 60.0


# ==================================================
# 4. 初始化各个模块
# ==================================================

# 装甲板检测器
detector = BlueArmorDetector()

# PnP位姿解算器
pnp_solver = PNPSolver(
    camera_matrix,
    dist_coeffs,
    ARMOR_WIDTH,
    ARMOR_HEIGHT
)

# 卡尔曼滤波器
tracker = KalmanTracker()

print("装甲板检测器初始化成功！")
print("PnP解算器初始化成功！")
print("卡尔曼滤波器初始化成功！")


# ==================================================
# 5. 打开摄像头
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("摄像头打开失败！")

# 设置摄像头分辨率
# 应与标定时使用的分辨率一致
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("摄像头启动成功！")
print("按 Q 键退出程序。")


# ==================================================
# 6. 初始化时间变量
# ==================================================

prev_frame_time = time.perf_counter()

fps = 0.0

# 记录连续未检测到目标的帧数
lost_frames = 0


# ==================================================
# 7. 实时检测主循环
# ==================================================

try:

    while True:

        # ------------------------------------------
        # 7.1 获取摄像头画面
        # ------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("摄像头画面读取失败！")
            break

        # ------------------------------------------
        # 7.2 计算实际帧间时间
        # ------------------------------------------

        current_frame_time = time.perf_counter()

        dt = current_frame_time - prev_frame_time

        prev_frame_time = current_frame_time

        dt = max(0.001, min(dt, 0.2))

        # ------------------------------------------
        # 7.3 装甲板检测
        # ------------------------------------------

        output, armors, mask = detector.detect(frame)

        # ------------------------------------------
        # 7.4 判断是否检测到装甲板
        # ------------------------------------------

        if len(armors) > 0:

            # 当前版本使用一个卡尔曼滤波器
            # 暂时选取第一个检测结果
            armor = armors[0]

            lost_frames = 0

            # --------------------------------------
            # 7.5 提取四个角点
            # --------------------------------------

            corners = np.asarray(
                armor["corners"],
                dtype=np.float32
            ).reshape(4, 2)

            # 绘制四个角点
            for point in corners:

                px, py = point.astype(int)

                cv2.circle(
                    output,
                    (px, py),
                    5,
                    (0, 255, 255),
                    -1
                )

            # 绘制装甲板四边形
            cv2.polylines(
                output,
                [corners.astype(np.int32)],
                True,
                (0, 255, 0),
                2
            )

            # --------------------------------------
            # 7.6 PnP三维位姿解算
            # --------------------------------------

            pose = pnp_solver.solve(corners)

            if pose is not None:

                x = pose["x"]
                y = pose["y"]
                z = pose["z"]

                distance = pose["distance"]

                # 目标必须位于相机前方
                if z > 0:

                    # ----------------------------------
                    # 7.7 卡尔曼滤波
                    # ----------------------------------

                    filtered = tracker.update(
                        [x, y, z],
                        dt
                    )

                    fx = filtered["x"]
                    fy = filtered["y"]
                    fz = filtered["z"]

                    # ----------------------------------
                    # 7.8 获取目标速度
                    # ----------------------------------

                    vx = filtered["vx"]
                    vy = filtered["vy"]
                    vz = filtered["vz"]

                    # ----------------------------------
                    # 7.9 预测未来0.1秒的位置
                    # ----------------------------------

                    predicted = tracker.predict_future(
                        0.1
                    )

                    px, py, pz = predicted

                    # ----------------------------------
                    # 7.10 显示PnP原始坐标
                    # ----------------------------------

                    cv2.putText(
                        output,
                        f"Raw XYZ: {x:.1f}, {y:.1f}, {z:.1f}",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------
                    # 7.11 显示距离
                    # ----------------------------------

                    cv2.putText(
                        output,
                        f"Distance: {distance:.1f} mm",
                        (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

                    # ----------------------------------
                    # 7.12 显示滤波后坐标
                    # ----------------------------------

                    cv2.putText(
                        output,
                        f"Filtered: {fx:.1f}, {fy:.1f}, {fz:.1f}",
                        (15, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 0),
                        2
                    )

                    # ----------------------------------
                    # 7.13 显示三维速度
                    # ----------------------------------

                    cv2.putText(
                        output,
                        f"Velocity: {vx:.1f}, {vy:.1f}, {vz:.1f}",
                        (15, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 0),
                        2
                    )

                    # ----------------------------------
                    # 7.14 显示预测位置
                    # ----------------------------------

                    cv2.putText(
                        output,
                        f"Predict 0.1s: {px:.1f}, {py:.1f}, {pz:.1f}",
                        (15, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 165, 255),
                        2
                    )

                else:

                    cv2.putText(
                        output,
                        "Invalid PnP depth",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            else:

                cv2.putText(
                    output,
                    "PnP Failed",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        else:

            # --------------------------------------
            # 7.15 未检测到目标
            # --------------------------------------

            lost_frames += 1

            cv2.putText(
                output,
                "No Armor Detected",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # 暂时不对丢失目标执行卡尔曼更新
            # 避免把不存在的测量值输入滤波器

        # ------------------------------------------
        # 7.16 显示检测数量
        # ------------------------------------------

        cv2.putText(
            output,
            f"Armors: {len(armors)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # ------------------------------------------
        # 7.17 显示FPS
        # ------------------------------------------

        fps = 0.9 * fps + 0.1 / dt

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ------------------------------------------
        # 7.18 显示目标丢失帧数
        # ------------------------------------------

        cv2.putText(
            output,
            f"Lost Frames: {lost_frames}",
            (15, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ------------------------------------------
        # 7.19 显示实时画面
        # ------------------------------------------

        cv2.imshow(
            "Live Armor Detection",
            output
        )

        cv2.imshow(
            "Blue Armor Mask",
            mask
        )

        # ------------------------------------------
        # 7.20 按Q退出
        # ------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    # ==================================================
    # 8. 释放资源
    # ==================================================

    cap.release()

    cv2.destroyAllWindows()

    print("摄像头已关闭！")
    print("程序正常退出！")