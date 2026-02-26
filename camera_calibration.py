# -*- coding: gbk -*-
import cv2
import numpy as np
import os
import glob
import time
import subprocess
import re

# ====================== Calibration Parameter Configuration ======================
# 棋盘格内角点数量（9列×6行，需匹配实际棋盘格）
CHESSBOARD_SIZE = (9, 6)  
# 棋盘格方格边长（根据实际方格大小修改此参数）
SQUARE_SIZE = 0.009        
# 摄像头设备节点
CAM_ID = "/dev/video0"
# 标定图像保存目录
CALIB_IMG_DIR = "calibration_images"
# 标定结果保存文件
CALIB_RESULT_FILE = "stereo_calib_params.npz"

# 摄像头分辨率参数
CAM_WIDTH = 2560
CAM_HEIGHT = 720
CAM_FPS = 30 
LEFT_WIDTH = CAM_WIDTH // 2 
RIGHT_WIDTH = CAM_WIDTH // 2
FRAME_HEIGHT = CAM_HEIGHT

# ====================== Helper Function: Create Directory ======================
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Create directory: {dir_path}")

# ====================== New Function: Read Camera Control Parameters (V4L2) ======================
def get_camera_controls(cam_device):
    """
    读取V4L2摄像头的所有可控参数（亮度、曝光、白平衡等）
    :param cam_device: 摄像头设备节点（如/dev/video0）
    :return: 字典格式的参数，key=参数名，value=参数值/状态
    """
    controls = {}
    try:
        # 调用v4l2-ctl --all获取所有参数
        result = subprocess.check_output(
            ["v4l2-ctl", "-d", cam_device, "--all"],
            stderr=subprocess.STDOUT,
            text=True
        )

        # 解析User Controls（用户可控参数）
        user_controls_pattern = re.compile(r"User Controls\n(.*?)\n\nCamera Controls", re.DOTALL)
        user_controls_match = user_controls_pattern.search(result)
        if user_controls_match:
            user_controls_str = user_controls_match.group(1)
            # 解析每行参数（如：brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=64）
            param_pattern = re.compile(r"(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s+:\s+min=(-?\d+) max=(\d+) step=(\d+) default=(\d+) value=(\d+)")
            # 布尔型参数（如：white_balance_automatic 0x0098090c (bool)   : default=1 value=0）
            bool_pattern = re.compile(r"(\w+)\s+0x[0-9a-f]+\s+\(bool\)\s+:\s+default=(\d+) value=(\d+)")
            # 菜单型参数（如：power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1 (50 Hz)）
            menu_pattern = re.compile(r"(\w+)\s+0x[0-9a-f]+\s+\(menu\)\s+:\s+min=(\d+) max=(\d+) default=(\d+) value=(\d+) \((.*?)\)")

            # 解析数值型参数
            for match in param_pattern.finditer(user_controls_str):
                name, type_, min_val, max_val, step, default, value = match.groups()
                controls[name] = {
                    "type": type_,
                    "min": int(min_val),
                    "max": int(max_val),
                    "step": int(step),
                    "default": int(default),
                    "current": int(value)
                }

            # 解析布尔型参数
            for match in bool_pattern.finditer(user_controls_str):
                name, default, value = match.groups()
                controls[name] = {
                    "type": "bool",
                    "default": bool(int(default)),
                    "current": bool(int(value))
                }

            # 解析菜单型参数
            for match in menu_pattern.finditer(user_controls_str):
                name, min_val, max_val, default, value, desc = match.groups()
                controls[name] = {
                    "type": "menu",
                    "min": int(min_val),
                    "max": int(max_val),
                    "default": int(default),
                    "current": int(value),
                    "current_desc": desc.strip()
                }

        # 解析Camera Controls（摄像头专用参数，如曝光模式）
        camera_controls_pattern = re.compile(r"Camera Controls\n(.*?)$", re.DOTALL)
        camera_controls_match = camera_controls_pattern.search(result)
        if camera_controls_match:
            camera_controls_str = camera_controls_match.group(1)
            # 解析曝光模式/曝光值（补充完整参数结构）
            exp_mode_pattern = re.compile(r"auto_exposure\s+0x[0-9a-f]+\s+\(menu\)\s+:\s+min=(\d+) max=(\d+) default=(\d+) value=(\d+) \((.*?)\)")
            exp_time_pattern = re.compile(r"exposure_time_absolute\s+0x[0-9a-f]+\s+\(int\)\s+:\s+min=(\d+) max=(\d+) step=(\d+) default=(\d+) value=(\d+)")
            
            exp_mode_match = exp_mode_pattern.search(camera_controls_str)
            if exp_mode_match:
                min_val, max_val, default, value, desc = exp_mode_match.groups()
                controls["auto_exposure"] = {
                    "type": "menu",
                    "min": int(min_val),
                    "max": int(max_val),
                    "default": int(default),
                    "current": int(value),
                    "current_mode": desc.strip()
                }
            
            exp_time_match = exp_time_pattern.search(camera_controls_str)
            if exp_time_match:
                min_val, max_val, step, default, value = exp_time_match.groups()
                controls["exposure_time_absolute"] = {
                    "type": "int",
                    "min": int(min_val),
                    "max": int(max_val),
                    "step": int(step),
                    "default": int(default),
                    "current": int(value)
                }

    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to read camera controls (v4l2-ctl error): {e.output}")
    except Exception as e:
        print(f"Warning: Parse camera controls failed: {str(e)}")

    return controls

# ====================== New Function: Print Camera Controls (Formatted) ======================
def print_camera_controls(controls):
    """格式化输出摄像头参数（修复KeyError，兼容所有参数结构）"""
    print("\n===== Camera Control Parameters (V4L2) =====")
    if not controls:
        print("No controllable parameters found!")
        return
    
    # 分类输出关键参数
    key_params = [
        "brightness", "contrast", "saturation", "hue",
        "white_balance_automatic", "white_balance_temperature",
        "power_line_frequency", "gamma", "sharpness",
        "backlight_compensation", "auto_exposure", "exposure_time_absolute"
    ]
    
    for param in key_params:
        if param not in controls:
            continue  # 跳过不存在的参数
        val = controls[param]
        
        # 按参数类型分别处理，避免KeyError
        if param == "white_balance_automatic":
            # 布尔型（自动白平衡）
            print(f"{param:30} : {'Auto' if val['current'] else 'Manual'} (default: {'Auto' if val['default'] else 'Manual'})")
        
        elif param == "auto_exposure":
            # 曝光模式（菜单型，补充了min/max/default）
            print(f"{param:30} : {val['current_mode']} (value: {val['current']}, min: {val['min']}, max: {val['max']}, default: {val['default']})")
        
        elif val["type"] == "bool":
            # 通用布尔型参数
            print(f"{param:30} : {val['current']} (default: {val['default']})")
        
        elif val["type"] == "menu":
            # 通用菜单型参数
            print(f"{param:30} : {val.get('current_desc', val['current'])} (value: {val['current']}, min: {val['min']}, max: {val['max']}, default: {val['default']})")
        
        elif val["type"] == "int":
            # 通用数值型参数
            print(f"{param:30} : {val['current']} (min: {val['min']}, max: {val['max']}, step: {val.get('step', '-')}, default: {val['default']})")
        
        else:
            # 其他未知类型，直接输出
            print(f"{param:30} : {val}")
    
    # 输出其他参数
    print("\nOther controls (raw format):")
    other_params = [k for k in controls.keys() if k not in key_params]
    for param in other_params:
        print(f"{param:30} : {controls[param]}")
    print("="*50)

# ====================== Step 1: Capture Calibration Image Pairs (Optimized) ======================
def capture_calibration_images():
    left_dir = os.path.join(CALIB_IMG_DIR, "left")
    right_dir = os.path.join(CALIB_IMG_DIR, "right")
    create_dir(left_dir)
    create_dir(right_dir)

    # 打开摄像头（V4L2接口）
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Failed to open camera! Check device node {CAM_ID}")
        return

    # 读取并输出摄像头控制参数
    print("\n===== Reading Camera Hardware Parameters =====")
    cam_controls = get_camera_controls(CAM_ID)
    print_camera_controls(cam_controls)

    # 参数设置顺序：先设分辨率/帧率，再设FourCC（很多摄像头要求此顺序）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    
    # 强制设置MJPG格式
    # fourcc_mjpg = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    # 尝试设置MJPG，输出是否成功
    # set_fourcc_ok = cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    # print(f"\nInfo: Set MJPG FourCC result: {set_fourcc_ok} (True=成功, False=接口返回失败但实际可能生效)")

    # 可选：关闭自动曝光/白平衡（按需开启）
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 50)
    # cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 6500)

    # 校验摄像头实际参数
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8*i) & 0xFF) for i in range(3,-1,-1)])
    
    print(f"\n===== Camera Parameter Check =====")
    print(f"Expected: {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS}fps, MJPG")
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
    display_width = int(CAM_WIDTH * display_scale)
    display_height = int(CAM_HEIGHT * display_scale)
    left_display_width = int(LEFT_WIDTH * display_scale)

    while True:
        start_time = time.time()
        ret, frame_combined = cap.read()
        if not ret or frame_combined is None:
            print("Warning: Failed to read camera frame, skip")
            continue

        # 检查帧尺寸是否符合预期
        if frame_combined.shape[1] != CAM_WIDTH or frame_combined.shape[0] != CAM_HEIGHT:
            print(f"Warning: Frame size mismatch! Expected {CAM_WIDTH}x{CAM_HEIGHT}, got {frame_combined.shape[1]}x{frame_combined.shape[0]}")
            continue

        # 分割左右摄像头图像
        frame_left = frame_combined[:, 0:LEFT_WIDTH]
        frame_right = frame_combined[:, RIGHT_WIDTH:CAM_WIDTH]

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
        
        # 显示图像窗口
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
        sleep_time = max(0, 1/CAM_FPS - elapsed_time)
        time.sleep(sleep_time)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture finished! Total {img_count} pairs saved to {CALIB_IMG_DIR}")

# ====================== Step 2: Stereo Camera Calibration (Optimized for Phone Chessboard) ======================
def calibrate_stereo_camera():
    # 生成棋盘格3D世界坐标
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # 转换为米单位

    # 初始化角点存储列表
    objpoints = []       # 3D世界坐标点
    imgpoints_left = []  # 左摄像头2D角点
    imgpoints_right = [] # 右摄像头2D角点

    # 读取标定图像对
    left_img_paths = sorted(glob.glob(os.path.join(CALIB_IMG_DIR, "left", "*.jpg")))
    right_img_paths = sorted(glob.glob(os.path.join(CALIB_IMG_DIR, "right", "*.jpg")))

    # 校验图像对数量
    if len(left_img_paths) != len(right_img_paths):
        print(f"Error: Left images ({len(left_img_paths)}) != Right images ({len(right_img_paths)})")
        return
    if len(left_img_paths) < 10:
        print(f"Error: Need at least 10 image pairs, current {len(left_img_paths)}")
        return

    # 初始化参数
    valid_count = 0
    img_size = None  
    # 角点检测增强参数
    corner_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    # 亚像素优化准则
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 遍历所有图像对
    for idx, (left_path, right_path) in enumerate(zip(left_img_paths, right_img_paths)):
        # 读取图像
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        if img_left is None or img_right is None:
            print(f"Warning: Skip {idx}th pair - image read failed")
            continue
        
        # 转换为灰度图
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # 从第一张有效图像获取实际尺寸
        if img_size is None:
            img_size = (gray_left.shape[1], gray_left.shape[0])  # (width, height)
            print(f"Detected actual image size: {img_size}")

        # 检测棋盘格角点
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, None, corner_flags)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, None, corner_flags)


        if ret_left and ret_right:
            # 亚像素级角点优化
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # 添加到标定数据集
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            valid_count += 1

            # 1. 在原始尺寸图像上绘制角点
            img_left_temp = img_left.copy()
            img_right_temp = img_right.copy()
            cv2.drawChessboardCorners(img_left_temp, CHESSBOARD_SIZE, corners_left, ret_left)
            cv2.drawChessboardCorners(img_right_temp, CHESSBOARD_SIZE, corners_right, ret_right)
            
            # 2. 缩放图像到640x480用于显示
            img_left_draw = cv2.resize(img_left_temp, (640, 480))
            img_right_draw = cv2.resize(img_right_temp, (640, 480))
            
            # 3. 显示绘制结果
            cv2.imshow("Left Corners (Phone Chessboard)", img_left_draw)
            cv2.imshow("Right Corners (Phone Chessboard)", img_right_draw)
            cv2.waitKey(200)

    # 关闭所有显示窗口
    cv2.destroyAllWindows()
    print(f"\nValid image pairs for calibration: {valid_count} (need ≥10)")
    
    # 校验有效图像对数量
    if valid_count < 10:
        print("Error: Insufficient valid pairs! Capture more clear images of phone chessboard.")
        return

    # 单目标定（左摄像头）
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None
    )
    # 计算左摄像头重投影误差
    error_l = 0
    for i in range(len(objpoints)):
        img_pts_l, _ = cv2.projectPoints(objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
        error_l += cv2.norm(imgpoints_left[i], img_pts_l, cv2.NORM_L2) / len(img_pts_l)
    error_l /= len(objpoints)

    # 单目标定（右摄像头）
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None
    )
    # 计算右摄像头重投影误差
    error_r = 0
    for i in range(len(objpoints)):
        img_pts_r, _ = cv2.projectPoints(objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
        error_r += cv2.norm(imgpoints_right[i], img_pts_r, cv2.NORM_L2) / len(img_pts_r)
    error_r /= len(objpoints)

    # 双目标定
    flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_SAME_FOCAL_LENGTH
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria_stereo, flags=flags
    )

    # 极线校正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )
    # 生成校正映射表
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

    # 保存所有标定参数
    np.savez(
        CALIB_RESULT_FILE,
        mtx_l=mtx_l, dist_l=dist_l,
        mtx_r=mtx_r, dist_r=dist_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
        img_size=img_size,
        baseline=abs(T[0][0]),
        square_size=SQUARE_SIZE,
        chessboard_size=CHESSBOARD_SIZE,
        reproj_error_l=error_l,
        reproj_error_r=error_r,
        reproj_error_stereo=ret_stereo
    )

    # 输出标定结果
    print("\n===== Calibration Results (Phone Chessboard: 10mm Square) =====")
    print(f"Chessboard size (inner corners): {CHESSBOARD_SIZE}")
    print(f"Square size: {SQUARE_SIZE*1000} mm")
    print(f"Left camera reprojection error: {error_l:.4f} (ideal <1)")
    print(f"Right camera reprojection error: {error_r:.4f} (ideal <1)")
    print(f"Stereo reprojection error: {ret_stereo:.4f} (ideal <1)")
    print(f"\nLeft camera intrinsic matrix:\n{mtx_l}")
    print(f"\nRight camera intrinsic matrix:\n{mtx_r}")
    print(f"\nRotation matrix R:\n{R}")
    print(f"\nTranslation vector T (meters):\n{T}")
    print(f"\nBaseline length B: {abs(T[0][0]):.4f} meters ({abs(T[0][0])*1000:.1f} mm)")
    print(f"\nAll parameters saved to: {CALIB_RESULT_FILE}")

# ====================== Main Function ======================
if __name__ == "__main__":
    # Step 1: 采集标定图像（先取消注释运行）
    # capture_calibration_images()
    
    # Step 2: 执行双目标定（采集完成后取消注释运行）
    calibrate_stereo_camera()