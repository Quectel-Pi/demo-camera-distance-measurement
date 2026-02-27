# -*- coding: gbk -*-
import threading
import time
import cv2
import numpy as np
from log_manager import LogManager
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from common import (
    STEREO_WIDTH, STEREO_HEIGHT, PREVIEW_WIDTH, PREVIEW_HEIGHT,
    CAMERA_DEV, CAPTURE_L_PATH, CAPTURE_R_PATH, g_state
)

class CameraSettings:
    """摄像头参数设置"""
    def __init__(self):
        self.brightness = 50
        self.contrast = 0
        self.saturation = 36
        self.hue = 0
        self.gamma = 100
        self.sharpness = 3
        self.backlight = 0
        self.exposure = 50
        self.auto_exposure = False
        self.white_balance = 4600
        self.auto_white_balance = True

class CameraManager:
    """摄像头管理类"""
    
    def __init__(self):
        self._preview_thread = None
        self._camera_settings = CameraSettings()
        
        self._initial_settings = CameraSettings()
        
        # 初始化缓冲帧
        g_state.buffer_frame1 = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        g_state.buffer_frame2 = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        
    def reset_parameters(self):
        """将相机参数恢复到初始状态"""
        self._camera_settings = CameraSettings()
        if g_state.preview_running:
            self.stop_preview()
            self.start_preview(g_state.current_cam)
        return self._camera_settings

    def stop_preview_and_reset_display(self, ui_preview_label):
        """停止相机预览并重置UI显示"""
        self.stop_preview()
        ui_preview_label.setText("Please click the buttons below to start the camera mode")
        # 清空帧缓存
        with g_state.frame_lock:
            g_state.raw_frame = None
            g_state.display_frame = None
        g_state.frame_ready = False
        # 记录日志
        from log_manager import LogManager
        LogManager.append_log("Preview stopped, resources released.", "INFO")
    
    def start_preview(self, cam_id: int):
        """启动预览线程"""
        self.stop_preview()
        g_state.current_cam = cam_id
        g_state.preview_running = True
        g_state.frame_ready = False
        g_state.has_click = False
        g_state.click_point = (-1, -1)
        
        self._preview_thread = threading.Thread(target=self._preview_thread_func, daemon=True)
        self._preview_thread.start()
        LogManager.append_log(f"Preview started for mode: {cam_id}","INFO")

    def stop_preview(self):
        """停止预览线程"""
        g_state.preview_running = False
        g_state.current_cam = 0
        
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)
        
        self._preview_thread = None
        
        # 清空帧缓存
        with g_state.frame_lock:
            g_state.raw_frame = None
            g_state.display_frame = None
        g_state.frame_ready = False
        
        # 重置测距状态
        g_state.has_click = False
        g_state.click_point = (-1, -1)
        with g_state.distance_lock:
            g_state.distance = 0.0
        
        LogManager.append_log("Preview stopped, resources released.","INFO")
        
    def _preview_thread_func(self):
        """预览线程函数"""
        if isinstance(CAMERA_DEV, str) and CAMERA_DEV.startswith("/dev/video"):
            cam_idx = int(CAMERA_DEV.replace("/dev/video", ""))
        else:
            cam_idx = int(CAMERA_DEV)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            LogManager.append_log(f"Error: Failed to open camera index {cam_idx}","ERROR")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STEREO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STEREO_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._apply_camera_settings(cap)
        
        frame_count = 0
        stat_frame_count = 0
        last_stat_time = time.time()
        
        while g_state.preview_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            
            if g_state.frame_ready:
                continue
            
            cam_id = g_state.current_cam
            
            write_idx = g_state.write_buffer_index
            target_buffer = g_state.buffer_frame1 if write_idx == 0 else g_state.buffer_frame2
            
            # 根据摄像头模式选择显示区域
            if cam_id == 1:  # 左摄像头
                frame_show = frame[:, :STEREO_WIDTH//2]
            elif cam_id == 2:  # 右摄像头
                frame_show = frame[:, STEREO_WIDTH//2:]
            elif cam_id == 0:  # 测距模式（显示左摄像头）
                frame_show = frame[:, :STEREO_WIDTH//2]
            else:
                continue
            
            frame_show = cv2.resize(frame_show, (PREVIEW_WIDTH, PREVIEW_HEIGHT), 
                                   interpolation=cv2.INTER_LINEAR)
            
            # 测距模式下绘制点击点
            if cam_id == 0:
                click_pt = g_state.click_point
                if g_state.has_click and click_pt[0] >= 0 and click_pt[1] >= 0:
                    cv2.circle(frame_show, (click_pt[0], click_pt[1]), 3, (0, 0, 255), -1)
            
            # 转换为RGB格式
            frame_show = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB)
            target_buffer[:] = frame_show
            
            with g_state.frame_lock:
                g_state.raw_frame = frame.copy()
            
            g_state.write_buffer_index = 1 - write_idx
            g_state.frame_ready = True
            
            frame_count += 1
            stat_frame_count += 1
            
            # 性能统计
            now = time.time()
            elapsed = now - last_stat_time
            if elapsed >= 5:
                LogManager.append_log(f"[Performance] FPS: {stat_frame_count / elapsed:.1f}, Total frames: {frame_count}","INFO")
                last_stat_time = now
                stat_frame_count = 0
        cap.release()
        LogManager.append_log(f"Camera released. Total frames: {frame_count}","INFO")
        
    def take_stereo_capture(self) -> tuple[bool, str]:
        """
        双目拍照
        Returns: (success, status_message)
        """
        print("Starting stereo capture...")
        self.stop_preview()
    
        try:
            if isinstance(CAMERA_DEV, str) and CAMERA_DEV.startswith("/dev/video"):
                cam_idx = int(CAMERA_DEV.replace("/dev/video", ""))
            else:
                cam_idx = int(CAMERA_DEV)
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        
            if not cap.isOpened():
                return False, f"Failed to open camera index {cam_idx} for capture"
        
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, STEREO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STEREO_HEIGHT)
            self._apply_camera_settings(cap)
        
            # 跳过前15帧等待稳定
            for _ in range(15):
                cap.read()
                time.sleep(0.066)
        
            # 捕获帧
            ret, frame = cap.read()
            cap.release()
        
            if not ret or frame is None:
                return False, "Capture failed: Empty frame"
         
            # 拆分左右帧
            left_frame = frame[:, :STEREO_WIDTH//2].copy()
            right_frame = frame[:, STEREO_WIDTH//2:].copy()
         
            # 缩放用于显示
            left_frame = cv2.resize(left_frame, (PREVIEW_WIDTH//2 - 4, PREVIEW_HEIGHT), 
                                    interpolation=cv2.INTER_LINEAR)
            right_frame = cv2.resize(right_frame, (PREVIEW_WIDTH//2 - 4, PREVIEW_HEIGHT), 
                                    interpolation=cv2.INTER_LINEAR)
        
            # 保存帧到文件
            cv2.imwrite(CAPTURE_L_PATH, left_frame)
            cv2.imwrite(CAPTURE_R_PATH, right_frame)
            print(f"Capture success: Saved to {CAPTURE_L_PATH} / {CAPTURE_R_PATH}")
        
            return True, "Capture completed successfully"
    
        except Exception as e:
            return False, f"Capture failed: {str(e)}"
    
    def update_preview_frame(self):
        """更新预览画面"""
        if g_state.preview_label is None:
            return
        if not g_state.frame_ready:
            return
        
        # 读取显示缓冲区
        write_idx = g_state.write_buffer_index
        read_idx = 1 - write_idx
        read_buffer = g_state.buffer_frame1 if read_idx == 0 else g_state.buffer_frame2
        
        if read_buffer is None:
            return
        
        # 转换为QImage
        h, w, ch = read_buffer.shape
        img = QImage(read_buffer.data, w, h, ch * w, QImage.Format.Format_RGB888)
        
        # 设置到标签
        g_state.preview_label.setPixmap(QPixmap.fromImage(img.copy()))
        g_state.frame_ready = False
        
    def save_camera_settings(self, brightness: int, contrast: int, saturation: int,
                            hue: int, gamma: int, sharpness: int, backlight: int,
                            exposure: int, auto_exposure: bool, white_balance: int,
                            auto_white_balance: bool):
        """保存摄像头设置"""
        self._camera_settings.brightness = brightness
        self._camera_settings.contrast = contrast
        self._camera_settings.saturation = saturation
        self._camera_settings.hue = hue
        self._camera_settings.gamma = gamma
        self._camera_settings.sharpness = sharpness
        self._camera_settings.backlight = backlight
        self._camera_settings.exposure = exposure
        self._camera_settings.auto_exposure = auto_exposure
        self._camera_settings.white_balance = white_balance
        self._camera_settings.auto_white_balance = auto_white_balance
        print("Camera settings saved")
        
    def get_camera_settings(self) -> CameraSettings:
        """获取摄像头设置"""
        return self._camera_settings
        
    def _apply_camera_settings(self, cap: cv2.VideoCapture):
        """应用摄像头设置"""
        settings = self._camera_settings
        cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, settings.contrast)
        cap.set(cv2.CAP_PROP_SATURATION, settings.saturation)
        cap.set(cv2.CAP_PROP_HUE, settings.hue)
        cap.set(cv2.CAP_PROP_GAMMA, settings.gamma)
        cap.set(cv2.CAP_PROP_SHARPNESS, settings.sharpness - 1)
        cap.set(cv2.CAP_PROP_BACKLIGHT, settings.backlight)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if settings.auto_exposure else 0.75)
        cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1.0 if settings.auto_white_balance else 0.0)
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, settings.white_balance)
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, settings.white_balance)
        print("Camera settings applied")
        
def mat_to_qimage(mat: np.ndarray) -> QImage:
    """将OpenCV Mat转换为QImage"""
    if mat is None:
       return QImage()
    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_mat.shape
    return QImage(rgb_mat.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()