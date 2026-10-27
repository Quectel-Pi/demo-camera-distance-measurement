# -*- coding: utf-8 -*-

import sys

import os



os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
# 允许触屏事件传递
os.environ["QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS"] = "rotate=0"
if "QT_QPA_PLATFORM" not in os.environ:
    if "WAYLAND_DISPLAY" in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    elif os.environ.get("XDG_SESSION_TYPE", "") == "x11":
        os.environ["QT_QPA_PLATFORM"] = "xcb"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)



from PySide6.QtWidgets import QApplication

from PySide6.QtCore import Qt

from ui_manager import UIManager





def main():

    print("Starting QuecPi Stereo Camera Application (Python)...")



    # 启用高DPI缩放
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )



    app = QApplication(sys.argv)

    from PySide6.QtGui import QFont
    screen = app.primaryScreen()
    if screen:
        dpi = screen.logicalDotsPerInch()
        print(f"Screen DPI: {dpi}, Resolution: {screen.size().width()}x{screen.size().height()}")
        # 根据DPI动态设置基础字体大小
        base_size = max(9, int(10 * dpi / 96))
        font = QFont()
        font.setPointSize(base_size)
        app.setFont(font)

    window = UIManager()
    window.show()

    return app.exec()





if __name__ == "__main__":

    sys.exit(main())