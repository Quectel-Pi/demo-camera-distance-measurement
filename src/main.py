# -*- coding: gbk -*-
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication
from ui_manager import UIManager


def main():
    print("Starting QuecPi Stereo Camera Application (Python)...")
    
    app = QApplication(sys.argv)
    
    
    window = UIManager()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    from PySide6.QtCore import Qt
    sys.exit(main())