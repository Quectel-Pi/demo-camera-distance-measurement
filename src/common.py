# -*- coding: utf-8 -*-

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

    注意：RK3576/RK3588 等平台会创建大量 rkisp/rkcif 虚拟 video 节点，

    这些节点会接受任意分辨率设置（例如 4096x2160）但实际无法采集，

    因此不能只依赖 cap.get() 返回的分辨率，必须真正尝试读取一帧。

    Returns:

        tuple: (camera_device, stereo_width, stereo_height) 或 (None, 0, 0)

    """

    print("=" * 50)

    print("Detecting stereo camera...")



    # 查找所有 video 设备（跳过 video-camera0 之类的符号链接）

    if os.name == 'nt':  # Windows

        devices = [f"\\\\?\\video{i}" for i in range(10)]

    else:  # Linux

        def _video_index(path):

            suffix = os.path.basename(path).replace("video", "")

            return int(suffix) if suffix.isdigit() else 99999

        devices = sorted(
            [d for d in glob.glob("/dev/video*") if not os.path.islink(d)],
            key=_video_index,
        )



    candidates = []

    for dev in devices:

        if os.name != 'nt' and not os.path.exists(dev):

            continue

        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2 if os.name != 'nt' else cv2.CAP_ANY)

        if not cap.isOpened():

            continue

        try:

            # 对 UVC 双目摄像头，MJPG 才能跑满高分辨率/高帧率

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)

            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:

                continue

            aspect_ratio = width / height

            is_stereo = aspect_ratio >= 1.8  # 双目通常是 16:9 或更宽

            # 关键：只有真正读到一帧，才说明这个节点是可用的采集设备

            ok, frame = cap.read()

            can_capture = bool(ok and frame is not None and getattr(frame, "size", 0) > 0)

            print(f"  {dev}: {width}x{height} (ratio: {aspect_ratio:.2f}) "
                  f"capture={'OK' if can_capture else 'FAIL'}"
                  f"{' [STEREO]' if is_stereo else ''}")

            if is_stereo and can_capture:

                # 用实际读到的帧尺寸修正（部分设备上报值与真实帧不一致）

                fh, fw = frame.shape[:2]

                if fw > 0 and fh > 0:

                    width, height = fw, fh

                candidates.append((dev, width, height))

        except Exception as exc:  # 单个设备异常不影响整体检测

            print(f"  {dev}: probe failed ({exc})")

        finally:

            cap.release()



    if candidates:

        best = max(candidates, key=lambda x: x[1] * x[2])

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

        self.click_point2 = (-1, -1)      # ★ 第二个点击点

        self.has_click2 = False           # ★ 第二个点是否已设置

        self.distance = 0.0

        self.distance_between = 0.0       # ★ 两点间距离

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