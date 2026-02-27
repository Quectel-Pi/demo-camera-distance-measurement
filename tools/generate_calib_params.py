# -*- coding: gbk -*-
"""
标定参数生成工具
用于读取拍摄的标定图像对，生成双目标定参数文件
"""
import cv2
import numpy as np
import os
import glob

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


# ====================== Camera Detection Function ======================
def get_image_size_from_calib_images():
    """
    从已采集的标定图像中获取图像尺寸
    
    Returns:
        tuple: (width, height) 或 None
    """
    left_img_paths = sorted(glob.glob(os.path.join(CALIB_IMG_DIR, "left", "*.jpg")))
    
    if not left_img_paths:
        return None
    
    # 读取第一张图像获取尺寸
    img = cv2.imread(left_img_paths[0])
    if img is None:
        return None
    
    height, width = img.shape[:2]
    return (width, height)

def calibrate_stereo_camera():
    """执行双目标定，生成标定参数文件"""
    
    # 从标定图像获取尺寸
    img_size = get_image_size_from_calib_images()
    
    if img_size is None:
        print("Error: No calibration images found!")
        print(f"Please run 'python tools/capture_calib_images.py' first to capture images.")
        return
    
    print(f"Detected image size from calibration images: {img_size[0]}x{img_size[1]}")
    
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
        print(f"Please run 'python tools/capture_calib_images.py' to capture more images.")
        return

    print(f"\nFound {len(left_img_paths)} image pairs for calibration.")

    # 初始化参数
    valid_count = 0
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
        else:
            print(f"Warning: Skip {idx}th pair - chessboard not detected in {'left' if not ret_left else 'right'} image")

    # 关闭所有显示窗口
    cv2.destroyAllWindows()
    print(f"\nValid image pairs for calibration: {valid_count} (need ≥10)")
    
    # 校验有效图像对数量
    if valid_count < 10:
        print("Error: Insufficient valid pairs! Capture more clear images of phone chessboard.")
        print("Tips:")
        print("  - Ensure the chessboard is fully visible in both left and right images")
        print("  - Avoid motion blur and reflections")
        print("  - Cover different angles and distances")
        return

    # 单目标定（左摄像头）
    print("\nCalibrating left camera...")
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
    print("Calibrating right camera...")
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
    print("Performing stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_SAME_FOCAL_LENGTH
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria_stereo, flags=flags
    )

    # 极线校正
    print("Computing rectification maps...")
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
    print("\n" + "=" * 50)
    print("Calibration Results")
    print("=" * 50)
    print(f"Image size: {img_size[0]}x{img_size[1]}")
    print(f"Chessboard size (inner corners): {CHESSBOARD_SIZE}")
    print(f"Square size: {SQUARE_SIZE*1000} mm")
    print(f"\nReprojection Errors:")
    print(f"  Left camera:  {error_l:.4f} (ideal <1)")
    print(f"  Right camera: {error_r:.4f} (ideal <1)")
    print(f"  Stereo:       {ret_stereo:.4f} (ideal <1)")
    print(f"\nIntrinsic Matrix (Left Camera):\n{mtx_l}")
    print(f"\nIntrinsic Matrix (Right Camera):\n{mtx_r}")
    print(f"\nRotation Matrix R:\n{R}")
    print(f"\nTranslation Vector T (meters):\n{T}")
    print(f"\nBaseline Length: {abs(T[0][0]):.4f} meters ({abs(T[0][0])*1000:.1f} mm)")
    print(f"\nCalibration file saved to: {CALIB_RESULT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    calibrate_stereo_camera()

