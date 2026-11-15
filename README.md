HEAD
# ArmorVisionProject
OpenCV armor detection and motion prediction system
# OpenCV 装甲板视觉识别与运动预测系统

## 1. 项目简介

本项目基于 Python 和 OpenCV 开发，实现了一个实时装甲板视觉识别与三维位姿估计系统。

系统通过摄像头获取实时图像，识别蓝色装甲板，提取装甲板四个角点，并结合相机标定参数进行 PnP 三维位姿解算。最后使用卡尔曼滤波对目标位置进行平滑处理，并预测目标未来的位置。

## 2. 主要功能

1. 相机标定：获取相机内参和畸变系数。
2. 图像去畸变：减小镜头畸变对图像的影响。
3. 装甲板识别：基于 HSV 颜色空间识别蓝色目标。
4. 四角点提取：提取装甲板的四个顶点。
5. PnP 位姿解算：计算目标相对于相机的三维位置。
6. 卡尔曼滤波：平滑目标位置数据。
7. 运动预测：根据目标运动状态预测未来位置。
8. 实时显示：通过摄像头窗口展示检测结果和相关数据。

## 3. 项目环境

- 操作系统：Windows
- Python：以本地 PyCharm 项目解释器为准
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- 开发工具：PyCharm

## 4. 项目目录

```text
ArmorVisionProject/
├── calibration/
│   ├── calibrate.py
│   ├── camera_params.npz
│   └── images/
├── detector/
│   ├── __init__.py
│   ├── armor_detector.py
│   └── images/
├── pose/
│   ├── __init__.py
│   └── pnp_solver.py
├── tracking/
│   ├── __init__.py
│   └── kalman_tracker.py
├── main.py
├── live_armor_test.py
├── requirements.txt
└── README.md
```

## 5. 环境安装

建议在 PyCharm 的项目 Terminal 中执行：

```bash
python -m pip install numpy opencv-python matplotlib pyyaml
```

## 6. 运行方法

1. 使用 PyCharm 打开项目。
2. 确认项目解释器已经安装所需依赖。
3. 确认 `calibration/camera_params.npz` 文件存在。
4. 连接摄像头。
5. 运行项目根目录下的 `main.py`。
6. 将蓝色装甲板放在摄像头前进行测试。
7. 按 Q 键退出程序。

## 7. 系统处理流程

```text
摄像头采集图像
       |
       v
图像预处理
       |
       v
蓝色装甲板检测
       |
       v
四角点提取
       |
       v
PnP 三维位姿解算
       |
       v
卡尔曼滤波
       |
       v
运动状态预测
       |
       v
实时显示结果
```

## 8. 运行结果

系统运行后可以显示：

- 装甲板识别状态
- 装甲板四角点
- 原始三维坐标
- 目标距离
- 卡尔曼滤波后的坐标
- 目标运动速度
- 未来 0.1 秒的位置预测

## 9. 注意事项

1. 摄像头编号默认为 0，可根据实际设备调整。
2. 摄像头分辨率应与标定时使用的分辨率保持一致。
3. PnP 解算使用的装甲板宽度和高度应替换为实际测量值。
4. `camera_params.npz` 是相机标定生成的参数文件，运行前需要准备好。
5. 当前系统主要完成蓝色目标识别，实际部署时需要根据目标和环境调整检测参数。
6. 目标距离和三维位置的准确性取决于相机标定质量、实际目标尺寸以及角点检测精度。

## 10. 项目总结

本项目完成了从相机图像采集、装甲板识别、角点提取、PnP 三维位姿解算，到卡尔曼滤波和运动预测的基本视觉处理流程。

通过模块化设计，将相机标定、目标检测、位姿解算和目标跟踪分别实现，便于后续调试、优化和功能扩展。
04e91be (Initial commit: OpenCV armor vision system)
