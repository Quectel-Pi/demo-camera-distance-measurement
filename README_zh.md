# 摄像头测距

[English](README.md) | 中文

本项目是一个基于Quectel pi H1单板电脑实现的双目摄像头测距应用，点击相机预览画面中的物体可以计算摄像头与物体在真实世界中的距离。

![测距结果显示](assets/test1.png)

## 功能特性
- 基于视差原理计算目标距离
   - 点击画面任意位置即可测距
   - 使用SGBM立体匹配算法
   - 支持标定参数优化测量精度
- 支持左/右摄像头单独预览
- 同时捕获左右摄像头的图像做对比
- 支持摄像头参数调整优化匹配结果
   - 亮度、对比度、饱和度调节
   - 曝光时间/自动曝光
   - 白平衡/自动白平衡
   - 伽马、锐度、背光补偿

---

## 环境配置

### 系统要求
- **操作系统**: Linux（使用V4L2接口）
- **Python版本**: 3.8+
- **OpenCV版本**: 4.8+
- **PySide6版本**: 6.5+
- **numpy版本**: 1.24+

### 依赖安装

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install PySide6>=6.5.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
```

### 系统依赖（Linux）
```bash
# V4L2工具（用于摄像头参数读取）
sudo apt-get install v4l-utils
```

---

## 项目代码结构简介

```

├── requirements.txt             # Python依赖列表
├── README.md                    # 项目说明文档
├── README_en.md                 # 英文说明文档
│
├── src/                         # 测距应用源码目录
|   ├── main.py                  # 主入口，启动测距应用 
│   ├── camera_manager.py        # 摄像头管理类，负责视频采集、预览、拍照
│   ├── ranging_calculator.py    # 测距计算器，基于视差计算距离
│   ├── ui_manager.py            # UI界面管理，PySide6 GUI实现
│   ├── common.py                # 公共配置、全局状态管理、摄像头自动检测
│   └── log_manager.py           # 日志管理类
│
├── tools/                       # 标定工具目录
│   ├── capture_calib_images.py  # 标定图像采集工具
│   ├── generate_calib_params.py # 标定参数生成工具
│   ├── stereo_calib_params.npz  # 标定参数文件（运行标定后生成）
│   └── calibration_images/      # 标定图像存储目录（运行时自动创建）
│       ├── left/                # 左摄像头标定图像
│       └── right/               # 右摄像头标定图像
│
└── assets/                      # 资源文件目录
    ├── test1.png                # 测距效果演示图
    └── pattern.png              # 标定使用的棋盘格图案
```

### 模块说明

| 文件 | 功能描述 |
|------|----------|
| `main.py` | 应用程序入口，初始化Qt应用并显示主窗口 |
| `src/camera_manager.py` | `CameraManager`类：摄像头预览线程、参数设置、双目拍照 |
| `src/ranging_calculator.py` | `RangingCalculator`类：加载标定参数、计算视差图、计算距离 |
| `src/ui_manager.py` | `UIManager`类：主界面、预览页、拍照页、测距页、设置页 |
| `src/common.py` | 全局配置（分辨率、设备路径）、`GlobalState`单例状态管理 |
| `src/log_manager.py` | `LogManager`类：日志收集与显示 |
| `tools/capture_calib_images.py` | 标定图像采集：拍摄棋盘格图像对并保存 |
| `tools/generate_calib_params.py` | 标定参数生成：读取图像对，计算双目标定参数 |

---

## 硬件要求

### 双目摄像头（规格不硬性要求，以下是本项目使用规格）
- **接口类型**: USB
- **分辨率要求**: 
  - 推荐分辨率: 2560×720（左右各1280×720）
  - 支持其他宽高比 ≥ 1.8 的双目摄像头
- **帧率**: 支持15fps以上
- **输出格式**: YUYV/MJPG

### 标定工具（推荐打印出来，手机展示也可以但是要避免反光）
- **棋盘格标定板**: 9×6 内角点
- **方格边长**: 约9mm（可用手机屏幕显示，具体长度根据实际情况调整）
- **标定图像**: 需要至少10对不同角度的图像对

### 运行环境
- **开发板/PC**: 支持USB摄像头
- **内存**: 建议2GB以上
- **GPU**: 不要求（纯CPU计算）

---

## 使用流程

### 完整测距流程

```
┌─────────────────────────────────────────────────────────────
│                     测距应用使用流程                         
├─────────────────────────────────────────────────────────────
│  第一步：标定图像采集                                         
│  ├── 运行 tools/capture_calib_images.py                     
│  ├── 按要求拍摄至少15对棋盘格图像                             
│  └── 图像保存至 tools/calibration_images/                    
│                                                             
│  第二步：标定参数生成                                         
│  ├── 运行 tools/generate_calib_params.py                    
│  ├── 自动读取标定图像并计算                                   
│  └── 生成 tools/stereo_calib_params.npz                     
│                                                              
│  第三步：运行测距应用                                        
|  ├──  cd src                                                 
│  ├── 运行 python3 main.py                                    
│  ├── 点击"Start Ranging Mode"进入测距模式                     
│  └── 在画面上点击目标位置获取距离                              
└─────────────────────────────────────────────────────────────
```

### 第一步：标定图像采集

**运行命令：**
```bash
python3 tools/capture_calib_images.py
```

**操作步骤：**
1. 程序启动后会显示左右摄像头预览画面
2. 将棋盘格标定板放置在摄像头前方不同位置和角度
3. 按 `s` 键保存当前图像对（建议采集15-20对）
4. 按 `q` 键退出采集程序

**注意事项：**
- 确保棋盘格完整出现在左右画面中
- 尽量覆盖不同角度、距离和位置
- 避免运动模糊和反光

### 第二步：标定参数生成

**运行命令：**
```bash
python3 tools/generate_calib_params.py
```

**程序自动执行：**
1. 读取 `tools/calibration_images/` 目录下的图像对
2. 检测棋盘格角点
3. 执行单目标定和双目标定
4. 计算重投影误差
5. 生成 `tools/stereo_calib_params.npz` 标定参数文件

**输出信息：**
- 左/右摄像头重投影误差（理想值 <1）
- 双目重投影误差
- 基线距离（Baseline）
- 内参矩阵和外参矩阵

### 第三步：运行测距应用

**运行命令：**
```bash
cd src
python3 main.py
```

**功能按钮说明：**
| 按钮 | 功能 |
|------|------|
| Left Camera Preview | 左摄像头单独预览 |
| Right Camera Preview | 右摄像头单独预览 |
| Take Left/Right Picture | 双目拍照（保存到/tmp/） |
| Start Ranging Mode | 进入测距模式 |

**测距操作：**
1. 点击 **"Start Ranging Mode"** 进入测距模式
2. 在预览画面上点击目标位置
3. 等待距离计算结果显示在顶部提示栏

**测距结果显示：**

![测距结果显示](assets/test1.png)

---

## 标定参数配置

如需修改标定参数，请编辑 `tools/capture_calib_images.py` 和 `tools/generate_calib_params.py` 中的配置：

```python
# 棋盘格内角点数量（列×行）
CHESSBOARD_SIZE = (9, 6)

# 棋盘格方格边长（米），根据实际测量时每个方格真实物理长度设置
SQUARE_SIZE = 0.009  # 9mm

```

---

## 技术原理

### 双目测距原理
```
距离 Z = (f × B) / d

其中:
- f: 焦距（像素）
- B: 基线距离（米）
- d: 视差（像素）
```

### 算法流程
1. 左右图像采集
2. 立体校正（基于标定参数）
3. SGBM视差计算
4. 视差转3D坐标
5. 提取目标点距离

---
