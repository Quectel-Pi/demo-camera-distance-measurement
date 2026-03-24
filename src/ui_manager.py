# -*- coding: gbk -*-
import os
import sys
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QMessageBox, QGroupBox,
    QSizePolicy, QGridLayout, QTextEdit, QScrollArea, QScrollBar,
    QScroller
)
from PySide6.QtCore import Qt, QTimer, QSize, QRect, QEvent, QPointF
from PySide6.QtGui import QFont, QPixmap, QImage, QMouseEvent, QTextCursor, QScreen
from common import PREVIEW_WIDTH, PREVIEW_HEIGHT, g_state, CAPTURE_L_PATH, CAPTURE_R_PATH
from camera_manager import CameraManager, mat_to_qimage
from ranging_calculator import RangingCalculator
from log_manager import LogManager
import cv2


class ScalableLabel(QLabel):
    """可自适应比例缩放的预览标签，支持双击切换全屏和点击测距"""
    # 类级别的回调函数
    _click_callback = None
    
    def __init__(self, aspect_ratio=16/9, parent=None):
        super().__init__(parent)
        self._aspect_ratio = aspect_ratio
        self._pixmap = None
        self._cached_size = QSize()
        self._cached_pixmap = None
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#333; color:white; border-radius:8px;")
        self.setAlignment(Qt.AlignCenter)
        # 启用鼠标追踪，确保能接收到鼠标事件
        self.setMouseTracking(True)
        # 启用触摸事件支持
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        # 触摸双击检测
        self._last_touch_time = 0
        self._last_touch_pos = QPointF()
        self._touch_double_click_threshold = 400  # 毫秒

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self._cached_size = QSize()
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if not self._pixmap:
            return
        current_size = self.size()
        if self._cached_size == current_size and self._cached_pixmap:
            super().setPixmap(self._cached_pixmap)
            return
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        scale = min(current_size.width() / img_w, current_size.height() / img_h)
        target_w = int(img_w * scale)
        target_h = int(img_h * scale)
        if target_w > 0 and target_h > 0:
            self._cached_pixmap = self._pixmap.scaled(
                target_w, target_h,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self._cached_size = current_size
            super().setPixmap(self._cached_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            self._update_scaled_pixmap()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return int(width / self._aspect_ratio)

    def get_scale_offset(self):
        if not self._pixmap:
            return 1.0, 0, 0
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        label_w = self.width()
        label_h = self.height()
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        offset_x = (label_w - disp_w) / 2
        offset_y = (label_h - disp_h) / 2
        return scale, offset_x, offset_y

    def mouseDoubleClickEvent(self, event):
        """双击切换全屏/还原预览"""
        if event.button() == Qt.LeftButton:
            if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), '_toggle_fullscreen_preview'):
                self.parent().parent()._toggle_fullscreen_preview()
        super().mouseDoubleClickEvent(event)
    
    def mousePressEvent(self, event):
        """鼠标点击触发测距"""
        if event.button() == Qt.LeftButton and self._click_callback:
            # 使用 localPos 获取本地坐标
            self._click_callback(event.localPos())
        super().mousePressEvent(event)
    
    def touchEvent(self, event):
        """触摸事件处理"""
        import time
        if event.type() == QEvent.Type.TouchBegin:
            touch_points = event.touchPoints()
            if touch_points:
                pos = touch_points[0].pos()
                current_time = int(time.time() * 1000)  # 毫秒
                
                # 检测双击：两次触摸间隔小于阈值且位置相近
                is_double_tap = (
                    current_time - self._last_touch_time < self._touch_double_click_threshold and
                    (pos - self._last_touch_pos).manhattanLength() < 50
                )
                
                if is_double_tap:
                    # 双击切换全屏
                    if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), '_toggle_fullscreen_preview'):
                        self.parent().parent()._toggle_fullscreen_preview()
                elif self._click_callback:
                    # 单击触发测距
                    self._click_callback(pos)
                
                # 记录本次触摸信息
                self._last_touch_time = current_time
                self._last_touch_pos = pos
            
            event.accept()
            return
        super().touchEvent(event)
    
    def event(self, event):
        """重写event方法确保能捕获触摸事件"""
        if event.type() == QEvent.Type.TouchBegin:
            self.touchEvent(event)
            return True
        return super().event(event)


class TouchScrollTextEdit(QTextEdit):
    """支持触摸滚动的日志文本框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 启用 QScroller 实现触摸滚动
        QScroller.grabGesture(self.viewport(), QScroller.ScrollerGestureType.TouchGesture)
        
    def event(self, event):
        # 让 QScroller 处理触摸事件
        return super().event(event)


class UIManager(QWidget):
    def setup_styles(self):
        self.setStyleSheet("""
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-size: 12px;
        }
        QLabel {
            background-color: transparent;
        }
        /* 滚动区域样式 */
        QScrollArea {
            background-color: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background-color: #313244;
            width: 14px;
            border-radius: 7px;
            margin: 2px;
        }
        QScrollBar::handle:vertical {
            background-color: #585b70;
            border-radius: 7px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #6c7086;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        #tips_label {
            background-color: #313244;
            color: #cdd6f4;
            font-size: 14px;
            border-radius: 8px;
            padding: 12px 15px;
            min-height: 40px;
            border: 2px solid #585b70;
            margin-bottom: 15px;
        }
        /* 统一的GroupBox样式 */
        QGroupBox {
            color: #89b4fa;
            font-weight: bold;
            border: 2px solid #585b70;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            background-color: #313244;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            margin-top: 5px;
            margin-left: 5px;
            background: transparent;
            padding: 0 8px;
        }
        /* Log区域内部样式 */
        #log_text_edit {
            background-color: #1a1a28;
            color: #cdd6f4;
            font-family: Consolas, Monaco, monospace;
            font-size: 11px;
            border: none;
            border-radius: 0;
            padding: 8px;
            min-height: 140px;
        }
        /* 输入框样式 */
        QLineEdit {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #585b70;
            border-radius: 4px;
            padding: 4px;
            width: 80px;
        }
        QLineEdit:focus {
            border-color: #89b4fa;
        }
        /* 复选框样式 */
        QCheckBox {
            color: #cdd6f4;
            spacing: 8px;
            background: transparent;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid #585b70;
        }
        QCheckBox::indicator:checked {
            background-color: #89b4fa;
            border-color: #89b4fa;
        }
        /* 功能按钮样式 */
        QPushButton[func_btn=true] {
            background-color: #89b4fa;
            color: #1e1e2e;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 13px;
            min-height: 35px;
        }
        QPushButton[func_btn=true]:hover { background-color: #74c7ec; }
        QPushButton[func_btn=true]:pressed { background-color: #585b70; }
        /* 相机控制按钮 */
        QPushButton[save_btn=true] {
            background-color: #a6e3a1;
            color: #1e1e2e;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 14px;
            min-height: 40px;
        }
        QPushButton[reset_btn=true] {
            background-color: #f9c74f;
            color: #1e1e2e;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 14px;
            min-height: 40px;
        }
        QPushButton[stop_btn=true] {
            background-color: #f38ba8;
            color: #1e1e2e;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 14px;
            min-height: 40px;
        }
        """)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_styles()
        self._camera_manager = CameraManager()
        self._ranging_calculator = RangingCalculator()
        
        # 状态变量，记录是否处于全屏预览
        self._is_fullscreen_preview = False
        
        # 窗口配置
        self.setWindowTitle("Camera Distance Measurement")
        
        # 自适应屏幕大小
        self._setup_window_size()

        # 初始化布局
        self._init_main_layout()
        g_state.preview_label = self.preview_label
        
        # 设置预览标签的点击回调
        ScalableLabel._click_callback = self._handle_preview_click

        # 初始化日志
        LogManager.append_log("Starting Camera Distance Mesurement Application", "INFO")

        # 加载标定参数
        npz_path = os.path.join(os.path.dirname(__file__), "..", "tools", "stereo_calib_params.npz")
        if self._ranging_calculator.load_calibration(npz_path):
            calib_log = f"Calibration loaded successfully! - Baseline: {self._ranging_calculator._baseline:.6f}m - Image size: {self._ranging_calculator._img_size[0]}x{self._ranging_calculator._img_size[1]}"
            LogManager.append_log(calib_log, "INFO")
            self.update_tips("Calibration parameters loaded successfully [Success]")
        else:
            LogManager.append_log("Calibration load failed, using non-calibration mode", "WARN")
            self.update_tips("Warning: Calibration parameters load failed [Warning]")

        # 定时器配置
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._camera_manager.update_preview_frame)
        self._preview_timer.start(50)

        self._distance_timer = QTimer(self)
        self._distance_timer.timeout.connect(self._update_distance_tips)
        self._distance_timer.start(50)

        # 日志刷新定时器（100ms一次）
        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._refresh_log)
        self._log_timer.start(100)

        LogManager.append_log("UI initialized successfully", "INFO")

    def _setup_window_size(self):
        """根据屏幕大小自适应设置窗口尺寸"""
        screen = self.screen()
        if screen is None:
            # 如果获取不到屏幕，使用默认值
            self.setMinimumSize(800, 600)
            self.resize(1200, 800)
            return
        
        # 获取屏幕可用区域（排除任务栏等）
        available_geometry = screen.availableGeometry()
        screen_width = available_geometry.width()
        screen_height = available_geometry.height()
        
        # 计算窗口大小（占屏幕的85%）
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)
        
        # 设置合理的最小尺寸（不小于屏幕的50%）
        min_width = min(800, int(screen_width * 0.5))
        min_height = min(600, int(screen_height * 0.5))
        self.setMinimumSize(min_width, min_height)
        
        # 设置窗口大小，但不超过屏幕可用区域
        self.resize(
            min(window_width, screen_width - 50),
            min(window_height, screen_height - 50)
        )
        
        # 居中显示窗口
        self._center_window(available_geometry)

    def _center_window(self, available_geometry: QRect):
        """将窗口居中显示"""
        window_width = self.width()
        window_height = self.height()
        x = available_geometry.x() + (available_geometry.width() - window_width) // 2
        y = available_geometry.y() + (available_geometry.height() - window_height) // 2
        self.move(x, y)

    def _init_main_layout(self):
        main_v = QVBoxLayout(self)
        main_v.setContentsMargins(20, 20, 20, 20)
        main_v.setSpacing(10)

        # 1. 顶部Tips栏
        self.tips_label = QLabel()
        self.tips_label.setObjectName("tips_label")
        self.tips_label.setTextFormat(Qt.RichText)
        self.tips_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_v.addWidget(self.tips_label)

        # 2. 下方左右分栏（保存为实例变量）
        self.bottom_h_layout = QHBoxLayout()
        self.bottom_h_layout.setSpacing(20)
        self.bottom_h_layout.setStretch(6, 4)

        # ===== 左栏：预览 + 功能按钮 =====
        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(15)

        # 预览区（支持双击和点击测距）
        self.preview_label = ScalableLabel(aspect_ratio=PREVIEW_WIDTH/PREVIEW_HEIGHT)
        self.preview_label.setText("Please click buttons to start camera mode")
        left_v.addWidget(self.preview_label, stretch=8)

        # 功能按钮网格
        btn_w = QWidget()
        btn_grid = QGridLayout(btn_w)
        btn_grid.setSpacing(10)

        self.btn_left = QPushButton("Left Camera Preview")
        self.btn_right = QPushButton("Right Camera Preview")
        self.btn_capture = QPushButton("Take Left/Right Picture")
        self.btn_ranging = QPushButton("Start Ranging Mode")
        for btn in [self.btn_left, self.btn_right, self.btn_capture, self.btn_ranging]:
            btn.setProperty("func_btn", True)

        btn_grid.addWidget(self.btn_left, 0, 0)
        btn_grid.addWidget(self.btn_right, 0, 1)
        btn_grid.addWidget(self.btn_capture, 1, 0)
        btn_grid.addWidget(self.btn_ranging, 1, 1)
        left_v.addWidget(btn_w, stretch=2)

        self.bottom_h_layout.addWidget(left_w)

        # ===== 右栏：日志区 + 相机控制区（包装在ScrollArea中） =====
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setFrameShape(QScrollArea.NoFrame)  # 无边框
   
        
        self.right_widget = QWidget()
        right_v = QVBoxLayout(self.right_widget)
        right_v.setContentsMargins(5, 0, 5, 0)
        right_v.setSpacing(10)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(0, 0, 0, 0)
        # 使用支持触摸滚动的自定义 QTextEdit
        self.log_edit = TouchScrollTextEdit()
        self.log_edit.setObjectName("log_text_edit")
        self.log_edit.setReadOnly(True)
        self.log_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 设置日志区的最小高度，防止被压缩太小
        self.log_edit.setMinimumHeight(120)
        log_layout.addWidget(self.log_edit)
        right_v.addWidget(log_group, stretch=3)  # 日志区占3份

        cam_ctrl_group = QGroupBox("Camera Control")
        cam_ctrl_v = QVBoxLayout(cam_ctrl_group)
        cam_ctrl_v.setSpacing(10)
        self._create_basic_params(cam_ctrl_v)
        self._create_advanced_params(cam_ctrl_v)
        self._create_ctrl_buttons(cam_ctrl_v)
        right_v.addWidget(cam_ctrl_group, stretch=2)  # 控制区占2份
        
        right_v.addStretch()  # 底部弹性空间
        
        self.right_scroll.setWidget(self.right_widget)
        self.bottom_h_layout.addWidget(self.right_scroll)
        main_v.addLayout(self.bottom_h_layout)

        # 绑定按钮事件
        self._bind_events()

    def _create_basic_params(self, parent_layout):
        s = self._camera_manager.get_camera_settings()
        self.bright = self._make_param_item("Brightness", str(s.brightness), "(-64~64)", parent_layout)
        self.contrast = self._make_param_item("Contrast", str(s.contrast), "(0~95)", parent_layout)
        self.saturation = self._make_param_item("Saturation", str(s.saturation), "(0~100)", parent_layout)
        self.hue = self._make_param_item("Hue", str(s.hue), "(-2000~2000)", parent_layout)
        self.gamma = self._make_param_item("Gamma", str(s.gamma), "(100~300)", parent_layout)
        self.sharpness = self._make_param_item("Sharpness", str(s.sharpness), "(1~7)", parent_layout)
        self.backlight = self._make_param_item("Backlight Comp", str(s.backlight), "(0/1)", parent_layout)

    def _create_advanced_params(self, parent_layout):
        s = self._camera_manager.get_camera_settings()

        # 曝光设置
        h_exp = QHBoxLayout()
        h_exp.addWidget(QLabel("Exposure Time"))
        self.auto_exp = QCheckBox("Auto")
        self.auto_exp.setChecked(s.auto_exposure)
        self.exp_val = QLineEdit(str(s.exposure))
        self.exp_val.setEnabled(not s.auto_exposure)
        self.auto_exp.toggled.connect(lambda c: self.exp_val.setEnabled(not c))
        h_exp.addWidget(self.auto_exp)
        h_exp.addWidget(self.exp_val)
        h_exp.addWidget(QLabel("(3~2047)"))
        h_exp.addStretch()
        parent_layout.addLayout(h_exp)

        # 白平衡设置
        h_wb = QHBoxLayout()
        h_wb.addWidget(QLabel("WB Temp"))
        self.auto_wb = QCheckBox("Auto")
        self.auto_wb.setChecked(s.auto_white_balance)
        self.wb_val = QLineEdit(str(s.white_balance))
        self.wb_val.setEnabled(not s.auto_white_balance)
        self.auto_wb.toggled.connect(lambda c: self.wb_val.setEnabled(not c))
        h_wb.addWidget(self.auto_wb)
        h_wb.addWidget(self.wb_val)
        h_wb.addWidget(QLabel("(2800~6500)"))
        h_wb.addStretch()
        parent_layout.addLayout(h_wb)

    def _make_param_item(self, name, val, tip, parent):
        h = QHBoxLayout()
        h.addWidget(QLabel(name))
        edit = QLineEdit(val)
        h.addWidget(edit)
        h.addWidget(QLabel(tip))
        h.addStretch()
        parent.addLayout(h)
        return edit

    def _create_ctrl_buttons(self, parent_layout):
        v_btn = QVBoxLayout()
        v_btn.setSpacing(8)

        self.btn_save = QPushButton("Save Parameters")
        self.btn_save.setProperty("save_btn", True)
        self.btn_save.clicked.connect(self._save_params)
        self.btn_save.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_reset = QPushButton("Reset Parameters")
        self.btn_reset.setProperty("reset_btn", True)
        self.btn_reset.clicked.connect(self._reset_params)
        self.btn_reset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setProperty("stop_btn", True)
        self.btn_stop.clicked.connect(self._stop_camera)
        self.btn_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        v_btn.addWidget(self.btn_save)
        v_btn.addWidget(self.btn_reset)
        v_btn.addWidget(self.btn_stop)
        parent_layout.addLayout(v_btn)

    def _bind_events(self):
        self.btn_left.clicked.connect(lambda: self._start_cam(1, "Left camera preview activated"))
        self.btn_right.clicked.connect(lambda: self._start_cam(2, "Right camera preview activated"))
        self.btn_ranging.clicked.connect(lambda: self._start_cam(0, "Ranging mode activated"))
        self.btn_capture.clicked.connect(self._capture_stereo)

    def _start_cam(self, mode, tip):
        self._camera_manager.start_preview(mode)
        LogManager.append_log(f"Preview started for mode: {mode}", "INFO")
        self.update_tips(f"Status: {tip} [Active]")

    def _capture_stereo(self):
        """双目拍照"""
        self.update_tips("Status: Capturing stereo frames... [Capture]")
        ok, msg = self._camera_manager.take_stereo_capture()
        if ok:
            l_img = cv2.imread(CAPTURE_L_PATH)
            r_img = cv2.imread(CAPTURE_R_PATH)
            if l_img is not None and r_img is not None:
                combined = cv2.hconcat([l_img, r_img])
                combined = cv2.resize(combined, (PREVIEW_WIDTH, PREVIEW_HEIGHT), cv2.INTER_LINEAR)
                q_img = mat_to_qimage(combined)
                self.preview_label.setPixmap(QPixmap.fromImage(q_img))
                log = f"Capture success: Saved to {CAPTURE_L_PATH} / {CAPTURE_R_PATH}"
                tip = f"Capture success [Success] | Preview shows combined image"
            else:
                log = f"Capture success but image read failed: {msg}"
                tip = f"Capture success [Success] (Image read failed)"
            LogManager.append_log(log, "INFO")
            self.update_tips(tip)
        else:
            LogManager.append_log(f"Capture failed: {msg}", "ERROR")
            self.update_tips(f"Capture failed [Failed]: {msg}")

    def _save_params(self):
        """保存相机参数"""
        try:
            # 读取并校验参数
            b = int(self.bright.text())
            c = int(self.contrast.text())
            s = int(self.saturation.text())
            h = int(self.hue.text())
            g = int(self.gamma.text())
            sh = int(self.sharpness.text())
            bl = int(self.backlight.text())
            exp = int(self.exp_val.text())
            wb = int(self.wb_val.text())
            auto_exp = self.auto_exp.isChecked()
            auto_wb = self.auto_wb.isChecked()

            # 校验范围
            if not (-64 <= b <= 64): raise ValueError("Brightness: -64~64")
            if not (0 <= c <= 95): raise ValueError("Contrast: 0~95")
            if not (0 <= s <= 100): raise ValueError("Saturation: 0~100")
            if not (-2000 <= h <= 2000): raise ValueError("Hue: -2000~2000")
            if not (100 <= g <= 300): raise ValueError("Gamma: 100~300")
            if not (1 <= sh <= 7): raise ValueError("Sharpness: 1~7")
            if not (0 <= bl <= 1): raise ValueError("Backlight Comp: 0/1")
            if not (3 <= exp <= 2047): raise ValueError("Exposure: 3~2047")
            if not (2800 <= wb <= 6500): raise ValueError("WB Temp: 2800~6500")

            # 保存参数
            self._camera_manager.save_camera_settings(b, c, s, h, g, sh, bl, exp, auto_exp, wb, auto_wb)
            QMessageBox.information(self, "Success", "Parameters saved!\nRestart camera to apply.")
            LogManager.append_log("Camera parameters saved", "INFO")
            self.update_tips("Status: Parameters saved [Success]")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            LogManager.append_log(f"Invalid param: {str(e)}", "WARN")
        except Exception as e:
            QMessageBox.warning(self, "Failed", f"Save error: {str(e)}")
            LogManager.append_log(f"Save failed: {str(e)}", "ERROR")

    def _reset_params(self):
        """重置相机参数到初始状态"""
        init_settings = self._camera_manager.reset_parameters()
        # 更新UI输入框
        self.bright.setText(str(init_settings.brightness))
        self.contrast.setText(str(init_settings.contrast))
        self.saturation.setText(str(init_settings.saturation))
        self.hue.setText(str(init_settings.hue))
        self.gamma.setText(str(init_settings.gamma))
        self.sharpness.setText(str(init_settings.sharpness))
        self.backlight.setText(str(init_settings.backlight))
        self.exp_val.setText(str(init_settings.exposure))
        self.auto_exp.setChecked(init_settings.auto_exposure)
        self.wb_val.setText(str(init_settings.white_balance))
        self.auto_wb.setChecked(init_settings.auto_white_balance)
        
        QMessageBox.information(self, "Success", "Parameters reset to initial!")
        LogManager.append_log("Camera parameters reset", "INFO")
        self.update_tips("Status: Parameters reset [Success]")

    def _stop_camera(self):
        """停止相机并重置预览区"""
        self._camera_manager.stop_preview_and_reset_display(self.preview_label)
        LogManager.append_log("Camera stopped, resources released", "INFO")
        self.update_tips("Status: Camera stopped [Stopped] | Click buttons to restart")

    def _toggle_fullscreen_preview(self):
        """切换预览区全屏/还原状态"""
        self._is_fullscreen_preview = not self._is_fullscreen_preview

        if self._is_fullscreen_preview:
            self.bottom_h_layout.setStretch(0, 100)
            self.bottom_h_layout.setStretch(1, 0)
            self.right_scroll.hide()
            self.preview_label.setStyleSheet("background:#333; color:white; border-radius:0px;")
        else:
            self.bottom_h_layout.setStretch(0, 6)
            self.bottom_h_layout.setStretch(1, 4)
            self.right_scroll.show()
            self.preview_label.setStyleSheet("background:#333; color:white; border-radius:8px;")

    def update_tips(self, text):
        self.tips_label.setText(text)

    def _update_distance_tips(self):
        """更新测距结果"""
        if g_state.current_cam != 0:
            return
        with g_state.distance_lock:
            d = g_state.distance
        if d > 0:
            tip = f"Status: Ranging mode active | Click preview to calculate distance<br>"
            tip += f"<span style='color:#f38ba8; font-size:16px; font-weight:bold;'>Measured distance: {d:.2f} meters</span>"
        else:
            tip = f"Status: Ranging mode active | Click preview to calculate distance<br>"
            tip += f"<span style='color:#f38ba8; font-size:16px; font-weight:bold;'>Measured distance: Invalid</span>"
        self.tips_label.setText(tip)

    def _refresh_log(self):
        """刷新日志显示，智能滚动到底部 + 日志等级变色"""
        new_log_lines = LogManager.get_log_lines()

        html = """
        <style>
            body {
                background-color: #1a1a28;
                color: #cdd6f4;
                font-family: Consolas, monospace;
                font-size: 11px;
                line-height: 1.3;
                margin: 0;
                padding: 0;
            }
            .log-info { color: #cdd6f4; }
            .log-warn { color: #f9c74f; }
            .log-error { color: #f38ba8; }
            .log-debug { color: #89b4fa; }
        </style>
        <body>
        """

        for line in new_log_lines:
            if "[ERROR]" in line:
                html += f"<div class='log-error'>{line}</div>"
            elif "[WARN]" in line:
                html += f"<div class='log-warn'>{line}</div>"
            elif "[DEBUG]" in line:
                html += f"<div class='log-debug'>{line}</div>"
            else:
                html += f"<div class='log-info'>{line}</div>"

        html += "</body>"

        if self.log_edit.toHtml() != html:
            # 先记录滚动条位置
            scroll = self.log_edit.verticalScrollBar()
            old_value = scroll.value()
            old_max = scroll.maximum()
            was_at_bottom = (old_value >= old_max - 5)

            # 更新内容
            self.log_edit.setHtml(html)

            if was_at_bottom:
                cursor = self.log_edit.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.log_edit.setTextCursor(cursor)
                self.log_edit.ensureCursorVisible()
            else:
                new_max = scroll.maximum()
                if new_max > 0:
                    new_value = int(old_value * new_max / max(old_max, 1))
                    scroll.setValue(new_value)

    def _handle_preview_click(self, pos):
        """处理预览区域的点击事件 - 触发测距
        
        Args:
            pos: QPointF 类型的位置对象，表示点击的本地坐标
        """
        if g_state.current_cam != 0 or g_state.preview_label is None:
            return
        
        # 检查是否在预览标签范围内
        if not (0 <= pos.x() <= self.preview_label.width() and 0 <= pos.y() <= self.preview_label.height()):
            return

        # 转换为图像原始坐标
        scale, ox, oy = self.preview_label.get_scale_offset()
        img_x = (pos.x() - ox) / scale
        img_y = (pos.y() - oy) / scale
        
        if not (0 <= img_x <= PREVIEW_WIDTH and 0 <= img_y <= PREVIEW_HEIGHT):
            return

        # 触发测距
        g_state.click_point = (int(img_x), int(img_y))
        g_state.has_click = True
        LogManager.append_log(f"Ranging click at: ({int(img_x)}, {int(img_y)})", "DEBUG")
        threading.Thread(target=self._ranging_calculator.calculate_distance, daemon=True).start()
