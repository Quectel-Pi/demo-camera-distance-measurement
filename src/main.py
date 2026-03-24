# -*- coding: gbk -*-
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("Starting QuecPi Stereo Camera Application (Python)...")
    
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    # Qt Virtual Keyboard 配置 - 必须在QApplication创建前设置
    os.environ["QT_IM_MODULE"] = "qtvirtualkeyboard"
    os.environ["QT_VIRTUALKEYBOARD_DESKTOP"] = "1"
    os.environ["QT_VIRTUALKEYBOARD_LAYOUT_PATH"] = ""
    
    # 键盘大小配置
    # 高度设为280像素，宽度将通过样式控制
    os.environ["QT_VIRTUALKEYBOARD_HEIGHT"] = "280"
    
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from ui_manager import UIManager
    
    app = QApplication(sys.argv)
    
    # 获取屏幕大小，动态计算键盘宽度（约为屏幕宽度的60%，对应左侧预览区域）
    screen = app.primaryScreen()
    if screen:
        screen_width = screen.availableGeometry().width()
        # 键盘宽度设为屏幕宽度的60%（与左侧预览区域宽度相近）
        keyboard_width = int(screen_width * 0.6)
        os.environ["QT_VIRTUALKEYBOARD_WIDTH"] = str(keyboard_width)
    
    # 设置应用全局样式
    app.setStyleSheet("""
        QWidget {
            font-size: 12px;
        }
    """)
    
    window = UIManager()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())