# -*- coding: gbk -*-
import os
import time
import threading
import numpy as np
import cv2
from log_manager import LogManager
from common import (
    STEREO_WIDTH, STEREO_HEIGHT, PREVIEW_WIDTH, PREVIEW_HEIGHT, g_state
)
IS_DEBUG = False
SAVE_DIR = os.path.join(os.path.dirname(__file__), "tmp_img")
class RangingCalculator:
    """双目测距计算器"""
    
    def __init__(self):
        self._baseline = 0.0
        self._img_size = (0, 0)
        self._is_calibrated = False
        
        # 标定参数
        self._mtx_l = None
        self._dist_l = None
        self._mtx_r = None
        self._dist_r = None
        self._map1x = None
        self._map1y = None
        self._map2x = None
        self._map2y = None
        self._Q = None
        
        # 创建调试目录
        if IS_DEBUG:
            self._create_dir_if_not_exist(SAVE_DIR)
            
    def load_calibration(self, npz_path: str) -> bool:
        """
        从NPZ文件加载标定参数
        
        Args:
            npz_path: NPZ文件路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(npz_path):
            LogManager.append_log(f"Error: Calibration file not found: {npz_path}","ERROR")
            return False
        
        try:
            data = np.load(npz_path)

            # 读取标定参数
            self._mtx_l = data.get('mtx_l')
            self._dist_l = data.get('dist_l')
            self._mtx_r = data.get('mtx_r')
            self._dist_r = data.get('dist_r')
            self._map1x = data.get('map1x')
            self._map1y = data.get('map1y')
            self._map2x = data.get('map2x')
            self._map2y = data.get('map2y')
            self._Q = data.get('Q')

            # 读取基线距
            self._baseline = float(data.get('baseline', 0.0))

            # 读取图像尺寸
            img_size = data.get('img_size')
            if img_size is not None:
                self._img_size = tuple(img_size)
            else:
                self._img_size = (0, 0)
                
            # 校验参数有效性
            if (self._mtx_l is None or self._map1x is None or 
                self._Q is None or self._img_size[0] == 0):
                LogManager.append_log("Error: Calibration parameters are invalid!","ERROR")
                return False
            
            self._is_calibrated = True
            LogManager.append_log("Calibration loaded successfully!","INFO")
            LogManager.append_log(f" - Baseline: {self._baseline} meters","INFO")
            LogManager.append_log(f" - Image size: {self._img_size[0]}x{self._img_size[1]}","INFO")
            return True
            
        except Exception as e:
            LogManager.append_log(f"Error loading calibration: {e}","ERROR")
            return False
            
    def calculate_distance(self):
        if not g_state.preview_running:
            LogManager.append_log("Error: Ranging failed - Camera is not running", "ERROR")
            return
        # 读取全局状态
        click_pt = g_state.click_point
        has_click = g_state.has_click
        
        # 校验输入有效性
        if not has_click or click_pt[0] < 0 or click_pt[1] < 0:
            with g_state.distance_lock:
                g_state.distance = 0.0
            LogManager.append_log("Error: Ranging failed - Invalid click point","ERROR")
            return
        
        with g_state.frame_lock:
            if g_state.raw_frame is None:
                with g_state.distance_lock:
                    g_state.distance = 0.0
                LogManager.append_log("Error: Ranging failed - Empty frame","ERROR")
                return
            raw_frame = g_state.raw_frame.copy()
            
        left_frame = raw_frame[:, :STEREO_WIDTH//2].copy()
        right_frame = raw_frame[:, STEREO_WIDTH//2:].copy()
        LogManager.append_log(f"Info: Captured left/right frames ({left_frame.shape[1]}x{left_frame.shape[0]})","INFO")
        
        # 计算原始点击点
        scale_x = (STEREO_WIDTH // 2) / PREVIEW_WIDTH
        scale_y = STEREO_HEIGHT / PREVIEW_HEIGHT
        raw_x = int(np.clip(click_pt[0] * scale_x, 0, STEREO_WIDTH // 2 - 1))
        raw_y = int(np.clip(click_pt[1] * scale_y, 0, STEREO_HEIGHT - 1))
        raw_point = (raw_x, raw_y)
        
        if IS_DEBUG:
            self._save_image_with_click_point(left_frame, raw_point, "raw_left")
            self._save_image_with_click_point(right_frame, raw_point, "raw_right")
        
        # 立体校正
        if self._is_calibrated:
            left_frame = cv2.remap(left_frame, self._map1x, self._map1y, cv2.INTER_LINEAR)
            right_frame = cv2.remap(right_frame, self._map2x, self._map2y, cv2.INTER_LINEAR)
            LogManager.append_log("Info: Frames undistorted with calibration params","INFO")
            if IS_DEBUG:
                self._save_image_with_click_point(left_frame, raw_point, "calib_left")
                self._save_image_with_click_point(right_frame, raw_point, "calib_right")
        else:
            LogManager.append_log("Warning: No calibration loaded - Using raw frames!","WARN")
        
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_left = clahe.apply(gray_left)
        gray_right = clahe.apply(gray_right)
        
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)
        gray_left = cv2.medianBlur(gray_left, 3)
        gray_right = cv2.medianBlur(gray_right, 3)
        
        # 保存灰度帧（仅debug模式）
        if IS_DEBUG:
            cv2.imwrite(self._get_timestamp_filename("gray_left", ".jpg"), gray_left)
            cv2.imwrite(self._get_timestamp_filename("gray_right", ".jpg"), gray_right)
        
        sgbm = self._init_stereo_sgbm()
        disparity_map = sgbm.compute(gray_left, gray_right)
        disparity_map = disparity_map.astype(np.float32) / 16.0
        
        # 保存视差图（仅debug模式）
        if IS_DEBUG:
            disparity_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.circle(disparity_vis, raw_point, 5, 255, -1)
            cv2.imwrite(self._get_timestamp_filename("disparity_map", ".jpg"), disparity_vis)
        
        disparity = 0.0
        valid_count = 0
        kernel = 5  # 5x5邻域
        
        for dy in range(-kernel//2, kernel//2 + 1):
            for dx in range(-kernel//2, kernel//2 + 1):
                x = raw_point[0] + dx
                y = raw_point[1] + dy
                if 0 <= x < gray_left.shape[1] and 0 <= y < gray_left.shape[0]:
                    d = disparity_map[y, x]
                    if d > 0.5:  # 过滤弱视差噪声
                        disparity += d
                        valid_count += 1
        
        if valid_count == 0:
            LogManager.append_log("Error: Ranging failed - No valid disparity points","ERROR")
            with g_state.distance_lock:
                g_state.distance = 0.0
            return
        
        disparity /= valid_count
        LogManager.append_log(f"Info: Average disparity: {disparity}","INFO")
        
        # 打印点击点处的视差值
        d = disparity_map[raw_point[1], raw_point[0]]
        LogManager.append_log(f"[Debug] Disparity at click point: {d}","DEBUG")
        
        distance = 0.0
        if self._is_calibrated and disparity > 0.5:
            xyz = cv2.reprojectImageTo3D(disparity_map, self._Q, False)
            point_3d = xyz[raw_point[1], raw_point[0]]
            LogManager.append_log(f"[Debug] 3D point: ({point_3d[0]}, {point_3d[1]}, {point_3d[2]})","DEBUG")
            
            z_3d = point_3d[2]
            if 0.01 < z_3d < 100.0:
                distance = z_3d
                LogManager.append_log(f"Success: Distance = {distance} meters (from 3D)","INFO")
            else:
                # Z不合理时用公式计算
                f = self._Q[2, 3]
                distance = (f * self._baseline) / disparity
                LogManager.append_log(f"Success: Distance = {distance} meters (from formula)","INFO")
        elif disparity > 0.5:
            # 无标定兼容模式
            fx = 695.0 if self._mtx_l is None else self._mtx_l[0, 0]
            baseline = 0.0735 if self._baseline <= 0 else self._baseline
            distance = (fx * baseline) / disparity
            LogManager.append_log(f"Success: Distance = {distance} meters (uncalibrated)","INFO")
        else:
            LogManager.append_log(f"Error: Invalid disparity ({disparity})","ERROR")
        
        # 更新距离
        with g_state.distance_lock:
            g_state.distance = distance
            
    def _init_stereo_sgbm(self) -> cv2.StereoSGBM:
        """初始化SGBM立体匹配器"""
        stereo = cv2.StereoSGBM_create(
              minDisparity=0,
              numDisparities=16*12,
              blockSize=11,
              P1=8*3*11*11,
              P2=32*3*11*11,
              disp12MaxDiff=1,
              uniquenessRatio=10,
              speckleWindowSize=100,
              speckleRange=32,
              mode=cv2.STEREO_SGBM_MODE_HH 
        )
        return stereo
    
    def _create_dir_if_not_exist(self, dir_path: str):
        if not IS_DEBUG:
            return
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            LogManager.append_log(f"[Debug] Created directory: {dir_path}","DEBUG")
    def _get_timestamp_filename(self, prefix: str, suffix: str) -> str:
        """生成带时间戳的文件名"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        return os.path.join(SAVE_DIR, f"{prefix}_{timestamp}_{ms:03d}{suffix}")
    
    def _save_image_with_click_point(self, img: np.ndarray, point: tuple, prefix: str):
        """保存带点击点标记的图片"""
        if not IS_DEBUG or img is None:
            return
        
        img_copy = img.copy()
        # 绘制红色实心圆
        cv2.circle(img_copy, point, 5, (0, 0, 255), -1)
        # 绘制绿色水平线
        cv2.line(img_copy, (0, point[1]), (img.shape[1], point[1]), (0, 255, 0), 1)
        # 绘制黄色邻域矩形
        cv2.rectangle(img_copy, (point[0] - 1, point[1] - 1), 
                     (point[0] + 1, point[1] + 1), (0, 255, 255), 1)
        
        filename = self._get_timestamp_filename(prefix, ".jpg")
        cv2.imwrite(filename, img_copy)
        LogManager.append_log(f"[Debug] Saved: {filename}","DEBUG")