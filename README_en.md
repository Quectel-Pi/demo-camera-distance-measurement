# Camera Ranging

## 1. Project Features

This project is a camera ranging application based on Quectel Pi H1 single-board computer, providing the following features:

### 1. Left/Right Camera Preview (Auxiliary function for checking left/right camera preview)
- Support separate preview for left/right cameras
- Real-time camera display
- Automatic detection and adaptation for stereo cameras

### 2. Left/Right Camera Picture (Auxiliary function for simultaneous left/right camera capture to check for misalignment)
- Simultaneously capture images from both left and right cameras
- Automatic left/right frame splitting
- Save to specified path

### 3. Binocular Ranging (Main function, uses left camera as preview, click on screen to measure distance)
- Calculate target distance based on disparity principle
- Click anywhere on the screen to measure distance
- Uses SGBM stereo matching algorithm
- Supports calibration parameters for improved measurement accuracy

### 4. Camera Settings (Auxiliary function for adjusting camera parameters)
- Brightness, contrast, saturation adjustment
- Exposure time / Auto exposure
- White balance / Auto white balance
- Gamma, sharpness, backlight compensation

---

## 2. Environment Configuration

### 2.1 System Requirements
- **Operating System**: Linux (using V4L2 interface)
- **Python Version**: 3.8+
- **OpenCV Version**: 4.8+
- **PySide6 Version**: 6.5+
- **NumPy Version**: 1.24+

### 2.2 Dependency Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install PySide6>=6.5.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
```

### 2.3 System Dependencies (Linux)
```bash
# V4L2 utilities (for camera parameter reading)
sudo apt-get install v4l-utils
```

---

## 3. Project Code Structure

```
PyCameraDemo/
├── main.py                 # Main entry point, launches GUI application
├── camera_manager.py       # Camera manager class, handles video capture, preview, and capture
├── camera_calibration.py   # Stereo calibration module, captures calibration images and computes parameters
├── ranging_calculator.py   # Ranging calculator, computes distance based on disparity
├── ui_manager.py           # UI manager, PySide6 GUI implementation
├── common.py               # Common configuration, global state management, auto camera detection
├── requirements.txt        # Python dependencies list
└── stereo_calib_params.npz # Calibration parameters file (generated after calibration)
```

### 3.1 Module Description

| File | Description |
|------|-------------|
| `main.py` | Application entry point, initializes Qt application and displays main window |
| `camera_manager.py` | `CameraManager` class: camera preview thread, parameter settings, stereo capture |
| `camera_calibration.py` | Calibration workflow: capture chessboard images → corner detection → stereo calibration → parameter saving |
| `ranging_calculator.py` | `RangingCalculator` class: load calibration parameters, compute disparity map, calculate distance |
| `ui_manager.py` | `UIManager` class: main page, preview page, capture page, ranging page, settings page |
| `common.py` | Global configuration (resolution, device path), `GlobalState` singleton state management |

---

## 4. Hardware Requirements

### 4.1 Stereo Camera (Specifications are not mandatory, below are the specs used in this project)
- **Interface Type**: USB
- **Resolution Requirements**:
  - Recommended: 2560×720 (1280×720 for each left/right camera)
  - Supports other stereo cameras with aspect ratio ≥ 1.8
- **Frame Rate**: 15fps or higher
- **Output Format**: YUYV/MJPG

### 4.2 Calibration Tool
- **Chessboard Pattern**: 9×6 inner corners
- **Square Size**: Approximately 9mm (can use phone screen display, adjust according to actual situation)

### 4.3 Runtime Environment
- **Development Board/PC**: USB camera support
- **Memory**: 2GB or more recommended
- **GPU**: Not required (pure CPU computation)

---

## 5. Usage

### 5.1 Camera Calibration (First-time Use)
```bash
# 1. Edit camera_calibration.py, uncomment capture_calibration_images()
# 2. Run calibration program, press 's' to save image pairs (at least 10 pairs)
# 3. Press 'q' to exit capture
# 4. Uncomment calibrate_stereo_camera() and run calibration computation
python camera_calibration.py
```

### 5.2 Run Main Program
```bash
python main.py
```


### 5.3 Ranging Operation
1. Click "Start Ranging Mode" to open the ranging page
2. Click on the target position on the screen
4. Wait for the distance calculation result to display

**Ranging Result Display:**

![Ranging Result](test1.png)

---

## 6. Technical Principles

### 6.1 Stereo Ranging Principle
```
Distance Z = (f × B) / d

Where:
- f: Focal length (pixels)
- B: Baseline distance (meters)
- d: Disparity (pixels)
```

### 6.2 Algorithm Workflow
1. Left/right image acquisition
2. Stereo rectification (based on calibration parameters)
3. SGBM disparity computation
4. Disparity to 3D coordinates conversion
5. Extract target point distance
