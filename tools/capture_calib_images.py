# -*- coding: gbk -*-
"""
标定图像采集工具
用于拍摄双目摄像头的标定图像对
"""
import cv2
import numpy as np
import os
import glob
import time
import subprocess
import re

# 获取当前脚本所在目录
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.dirname(TOOLS_DIR)

# ====================== Calibration Parameter Configuration ======================
# 棋盘格内角点数量（9列×6行，需匹配实际棋盘格）
CHESSBOARD_SIZE = (9, 6)  
# 棋盘格方格边长（根据实际方格大小修改此参数）
SQUARE_SIZE = 0.009        
# 标定图像保存目录
CALIB_IMG_DIR = os.path.join(TOOLS_DIR, "calibration_images")
# 标定结果保存文件
CALIB_RESULT_FILE = os.path.join(TOOLS_DIR, "stereo_calib_params.npz")


def detect_stereo_camera():
    """
    检测双目摄像头
    
    Returns:
        tuple: (camera_device, stereo_width, stereo_height) 或 (None, 0, 0)
    """
    print("=" * 50)
    print("Detecting stereo camera...")
    
    # 查找所有 video 设备
    if os.name == 'nt':  # Windows
        devices = [f"\\\\?\\video{i}" for i in range(10)]
    else:  # Linux
        devices = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")))
    
    stereo_devices = []
    
    for dev in devices:
        if os.name != 'nt' and not os.path.exists(dev):
            continue
            
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2 if os.name != 'nt' else cv2.CAP_ANY)
        if not cap.isOpened():
            continue
        
        try:
            # 获取支持的最大分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width > 0 and height > 0:
                # 检查是否是双目摄像头（宽度是高度的2倍以上）
                aspect_ratio = width / height
                is_stereo = aspect_ratio >= 1.8  # 双目通常是 16:9 或更宽
                
                print(f"  {dev}: {width}x{height} (ratio: {aspect_ratio:.2f}) {'[STEREO]' if is_stereo else ''}")
                
                if is_stereo:
                    stereo_devices.append((dev, width, height))
        finally:
            cap.release()
    
    if stereo_devices:
        # 选择分辨率最高的设备
        best = max(stereo_devices, key=lambda x: x[1] * x[2])
        print(f"\nSelected stereo camera: {best[0]} ({best[1]}x{best[2]})")
        print("=" * 50)
        return best
    else:
        print("\nNo stereo camera detected!")
        print("=" * 50)
        return None, 0, 0


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Create directory: {dir_path}")


def capture_calibration_images():
    """采集双目摄像头的标定图像对"""
    
    # 检测双目摄像头
    cam_dev, cam_width, cam_height = detect_stereo_camera()
    
    if cam_dev is None:
        print("Error: No stereo camera detected! Please check camera connection.")
        return
    
    # 设置摄像头参数
    cam_fps = 30
    left_width = cam_width // 2
    right_width = cam_width // 2
    
    left_dir = os.path.join(CALIB_IMG_DIR, "left")
    right_dir = os.path.join(CALIB_IMG_DIR, "right")
    create_dir(left_dir)
    create_dir(right_dir)

    # 打开摄像头（V4L2接口）
    cap = cv2.VideoCapture(cam_dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Failed to open camera! Check device node {cam_dev}")
        return

    # 参数设置顺序：先设分辨率/帧率，再设FourCC
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)
    
    # 强制设置YUYV格式
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    # 校验摄像头实际参数
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8*i) & 0xFF) for i in range(3,-1,-1)])
    
    print(f"\n===== Camera Parameter Check =====")
    print(f"Expected: {cam_width}x{cam_height} @ {cam_fps}fps, YUYV")
    print(f"Actual: {actual_width}x{actual_height} @ {actual_fps}fps, FourCC: {fourcc_str}")

    print("Clearing camera buffer...")
    for _ in range(10):
        cap.read()
    time.sleep(0.1)

    img_count = 0
    print("\n===== Start Calibration Image Capture =====")
    print("1. Place phone chessboard in front of camera (adjust angle/distance)")
    print("2. Press 's' to save image pair (at least 15 pairs required)")
    print("3. Press 'q' to exit capture")

    # 缩放显示参数
    display_scale = 0.5 
    display_width = int(cam_width * display_scale)
    display_height = int(cam_height * display_scale)
    left_display_width = int(left_width * display_scale)

    while True:
        start_time = time.time()
        ret, frame_combined = cap.read()
        if not ret or frame_combined is None:
            print("Warning: Failed to read camera frame, skip")
            continue

        # 检查帧尺寸是否符合预期
        if frame_combined.shape[1] != cam_width or frame_combined.shape[0] != cam_height:
            print(f"Warning: Frame size mismatch! Expected {cam_width}x{cam_height}, got {frame_combined.shape[1]}x{frame_combined.shape[0]}")
            continue

        # 分割左右摄像头图像
        frame_left = frame_combined[:, 0:left_width]
        frame_right = frame_combined[:, right_width:cam_width]

        # 检查分割后的帧非空
        if frame_left.size == 0 or frame_right.size == 0:
            print("Warning: Split frame is empty, skip")
            continue

        # 缩放显示图像
        frame_left_display = cv2.resize(frame_left, (left_display_width, display_height))
        frame_right_display = cv2.resize(frame_right, (left_display_width, display_height))

        # 添加提示文字
        cv2.putText(frame_left_display, f"Captured: {img_count} | 's' save | 'q' exit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_right_display, "Right Camera (Phone Chessboard)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Left Camera (Scaled)", frame_left_display)
        cv2.imshow("Right Camera (Scaled)", frame_right_display)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 保存图像对
            left_img_path = os.path.join(left_dir, f"left_{img_count:03d}.jpg")
            right_img_path = os.path.join(right_dir, f"right_{img_count:03d}.jpg")
            cv2.imwrite(left_img_path, frame_left)
            cv2.imwrite(right_img_path, frame_right)
            print(f"Saved {img_count+1}th pair: {left_img_path} | {right_img_path}")
            img_count += 1
        elif key == ord('q'):
            break

        # 控制帧率
        elapsed_time = time.time() - start_time
        sleep_time = max(0, 1/cam_fps - elapsed_time)
        time.sleep(sleep_time)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture finished! Total {img_count} pairs saved to {CALIB_IMG_DIR}")

if __name__ == "__main__":
    capture_calibration_images()

