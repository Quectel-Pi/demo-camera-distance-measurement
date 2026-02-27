# -*- coding: gbk -*-
import os
import glob
import cv2
import threading
import numpy as np

# 摄像头参数配置
STEREO_WIDTH = 2560
STEREO_HEIGHT = 720
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360
CAPTURE_L_PATH = "/tmp/capture_L.jpg"
CAPTURE_R_PATH = "/tmp/capture_R.jpg"


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
            # 获取支持的分辨率
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
   
        best = max(stereo_devices, key=lambda x: x[1] * x[2])
        print(f"\nSelected stereo camera: {best[0]} ({best[1]}x{best[2]})")
        print("=" * 50)
        return best
    else:
        print("\nNo stereo camera detected!")
        print("=" * 50)
        return None, 0, 0


CAMERA_DEV, DETECTED_WIDTH, DETECTED_HEIGHT = detect_stereo_camera()

# 如果检测到摄像头，更新分辨率
if CAMERA_DEV and DETECTED_WIDTH > 0:
    STEREO_WIDTH = DETECTED_WIDTH
    STEREO_HEIGHT = DETECTED_HEIGHT

    PREVIEW_WIDTH = STEREO_WIDTH // 4
    PREVIEW_HEIGHT = STEREO_HEIGHT // 2
else:
    # 使用默认设备
    CAMERA_DEV = "/dev/video0" if os.name != 'nt' else 0
    print(f"Warning: Using default camera device: {CAMERA_DEV}")
class GlobalState:
    """全局状态管理类"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_state()
        return cls._instance
    
    def _init_state(self):
        self.preview_running = False
        self.current_cam = 0  # 0:测距 1:左摄像头 2:右摄像头
        self.frame_lock = threading.Lock()
        self.raw_frame = None
        self.preview_label = None
        
        # 测距相关
        self.has_click = False
        self.click_point = (-1, -1)
        self.distance = 0.0
        self.distance_lock = threading.Lock()
        
        # 显示帧相关
        self.frame_ready = False
        self.display_frame = None
        
        # 缓冲帧
        self.buffer_frame1 = None
        self.buffer_frame2 = None
        self.write_buffer_index = 0

# 全局状态实例
g_state = GlobalState()