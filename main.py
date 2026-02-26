# -*- coding: gbk -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from ui_manager import UIManager


def main():
    """主函数"""
    print("Starting QuecPi Stereo Camera Application (Python)...")
    
    app = QApplication(sys.argv)
    
    
    window = UIManager()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    from PySide6.QtCore import Qt
    sys.exit(main())