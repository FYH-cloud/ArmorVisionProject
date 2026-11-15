import cv2
import numpy as np
import time
import os

from detector.armor_detector import BlueArmorDetector
from pose.pnp_solver import PNPSolver
from tracking.kalman_tracker import KalmanTracker


# ==========================================
# 1. 基本参数配置
# ==========================================

CAMERA_ID = 0

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 注意：这里是演示尺寸，必须替换成实际装甲板的宽和高
ARMOR_WIDTH = 160.0
ARMOR_HEIGHT = 60.0

# 未来位置预测时间（秒）
PREDICT_TIME = 0.1


# ==========================================
# 2. 加载相机标定参数
# ==========================================

def load_camera_parameters():

    possible_paths = [
        os.path.join("calibration", "camera_params.npz"),
        "camera_params.npz"
    ]

    for path in possible_paths:

        if os.path.exists(path):

            data = np.load(path)

            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]

            print("相机参数加载成功！")
            print("参数文件：", path)
            print("Camera Matrix:")
            print(camera_matrix)

            return camera_matrix, dist_coeffs

    raise FileNotFoundError(
        "没有找到 camera_params.npz，请检查标定文件位置！"
    )


# ==========================================
# 3. 绘制文字信息
# ==========================================

def draw_text(frame, text, position, color=(0, 255, 0)):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


# ==========================================
# 4. 主程序
# ==========================================

def main():

    print("================================")
    print("       OpenCV 装甲板视觉系统")
    print("================================")

    # 加载相机参数
    camera_matrix, dist_coeffs = load_camera_parameters()

    # 创建装甲板检测器
    detector = BlueArmorDetector()

    # 创建 PnP 解算器
    pnp_solver = PNPSolver(
        camera_matrix,
        dist_coeffs,
        ARMOR_WIDTH,
        ARMOR_HEIGHT
    )

    # 创建卡尔曼滤波器
    tracker = KalmanTracker()

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("摄像头打开失败！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    print("摄像头启动成功！")
    print("按 Q 键退出程序。")

    previous_time = time.perf_counter()

    lost_frames = 0

    # ======================================
    # 5. 实时视频处理循环
    # ======================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("摄像头图像读取失败！")
                break

            current_time = time.perf_counter()

            dt = current_time - previous_time
            previous_time = current_time

            dt = max(0.001, min(dt, 0.2))

            # 计算 FPS
            fps = 1.0 / dt

            # ==================================
            # 6. 装甲板检测
            # ==================================

            output, armors, mask = detector.detect(frame)

            draw_text(
                output,
                "Armor Vision System",
                (10, 25),
                (0, 255, 255)
            )

            draw_text(
                output,
                "FPS: {:.1f}".format(fps),
                (10, 50)
            )

            # ==================================
            # 7. 检测到装甲板
            # ==================================

            if len(armors) > 0:

                lost_frames = 0

                armor = armors[0]

                corners = np.asarray(
                    armor["corners"],
                    dtype=np.float32
                ).reshape(4, 2)

                # 绘制四个角点
                for point in corners:

                    x, y = point.astype(int)

                    cv2.circle(
                        output,
                        (x, y),
                        5,
                        (0, 255, 255),
                        -1
                    )

                # 绘制装甲板轮廓
                cv2.polylines(
                    output,
                    [corners.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )

                draw_text(
                    output,
                    "Armor Detected",
                    (10, 80),
                    (0, 255, 0)
                )

                # ==================================
                # 8. PnP 三维位姿解算
                # ==================================

                result = pnp_solver.solve(corners)

                if result is not None:

                    x = result["x"]
                    y = result["y"]
                    z = result["z"]

                    distance = result["distance"]

                    # 只处理相机前方的有效目标
                    if z > 0:

                        # ==================================
                        # 9. 卡尔曼滤波
                        # ==================================

                        filtered = tracker.update(
                            [x, y, z],
                            dt
                        )

                        fx = filtered["x"]
                        fy = filtered["y"]
                        fz = filtered["z"]

                        vx = filtered["vx"]
                        vy = filtered["vy"]
                        vz = filtered["vz"]

                        # ==================================
                        # 10. 未来位置预测
                        # ==================================

                        prediction =tracker.predict_future(PREDICT_TIME)


                        prediction = np.asarray(prediction).reshape(-1)


                        px = float(prediction[0])
                        py = float(prediction[1])
                        pz = float(prediction[2])

                        # ==================================
                        # 11. 显示 PnP 结果
                        # ==================================

                        draw_text(
                            output,
                            "Raw XYZ: {:.1f}, {:.1f}, {:.1f} mm".format(
                                x, y, z
                            ),
                            (10, 110)
                        )

                        draw_text(
                            output,
                            "Distance: {:.1f} mm".format(distance),
                            (10, 140)
                        )

                        # ==================================
                        # 12. 显示滤波结果
                        # ==================================

                        draw_text(
                            output,
                            "Filtered: {:.1f}, {:.1f}, {:.1f}".format(
                                fx, fy, fz
                            ),
                            (10, 170),
                            (255, 255, 0)
                        )

                        # ==================================
                        # 13. 显示运动速度
                        # ==================================

                        draw_text(
                            output,
                            "Velocity: {:.1f}, {:.1f}, {:.1f}".format(
                                vx, vy, vz
                            ),
                            (10, 200),
                            (255, 255, 0)
                        )

                        # ==================================
                        # 14. 显示预测位置
                        # ==================================

                        draw_text(
                            output,
                            "Predict 0.1s: {:.1f}, {:.1f}, {:.1f}".format(
                                px, py, pz
                            ),
                            (10, 230),
                            (0, 165, 255)
                        )

                        # 打印终端数据
                        print(
                            "Raw:",
                            round(x, 2),
                            round(y, 2),
                            round(z, 2),
                            "| Filtered:",
                            round(fx, 2),
                            round(fy, 2),
                            round(fz, 2)
                        )

            else:

                lost_frames += 1

                draw_text(
                    output,
                    "No Armor Detected",
                    (10, 80),
                    (0, 0, 255)
                )

            # ==================================
            # 15. 显示目标丢失帧数
            # ==================================

            draw_text(
                output,
                "Lost Frames: {}".format(lost_frames),
                (10, 260),
                (0, 255, 255)
            )

            # 显示画面
            cv2.imshow(
                "OpenCV Armor Vision System",
                output
            )

            # 显示蓝色检测掩膜
            cv2.imshow(
                "Blue Armor Mask",
                mask
            )

            # 按 Q 退出
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                break

    except KeyboardInterrupt:

        print("用户中断程序！")

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("摄像头已关闭！")
        print("程序运行结束！")


# ==========================================
# 16. 程序入口
# ==========================================

if __name__ == "__main__":

    main()