# Lerobot Controlled by hand

[![License](https://img.shields.io/github/license/ABMI-software/hand_controlLerobot)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)

## Overview
A robot arm teleoperation system based on hand tracking via webcam. Lerobot Control enables intuitive control of the SO-101 robot through natural hand gestures, bridging the gap between human motion and robotic control with a seamless real-time interface.

### Key Benefits
- 🎯 Intuitive Control: Natural hand movements translate directly to robot actions
- ⚡ Real-time Response: Minimal latency between gesture recognition and robot movement
- 🔄 Flexible Tracking: Multiple tracking models available for different use cases
- 🛠 Customizable: Adjustable sensitivity and control parameters

### What's New (Nov 17, 2025)
- **Headless & CI-friendly:** The repo now supports running on headless systems (WSL/CI) — see `scripts/run_poke_motor.sh` and the new `TROUBLESHOOTING.md` for env hints.
- **Clear camera handling:** Camera input accepts integer indices or file/device paths; code now fails with a clear, actionable message when no camera is present, and suggests using a video file for testing.
- **Lazy kinematics import:** The `pinocchio` dependency is loaded lazily — the app no longer fails at startup if `pinocchio` is missing (only required when using URDF/IK features).
- **Improved error messages & docs:** Helpful guidance added for common issues and step-by-step troubleshooting in `TROUBLESHOOTING.md`.
- **Unit tests:** New utility tests added for core mapping/clamping routines (`tests/test_utils.py`).


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features
- Real-time hand tracking and gesture recognition
- Control of robot joints and gripper
- Support for multiple tracking models (Wilor, MediaPipe)
- Object detection using YOLOv8 with Arducam
- Dual camera system (Arducam + webcam)
- Interactive mode selection (auto-pick/teleoperation)
- Adjustable parameters for sensitivity and control
- Safety limits and emergency stops
- Gesture recording and playback capabilities

## Project Structure
```
# Repository layout (top-level)
.
├── assets/                         # Images, models and other assets
├── hand_teleop/                    # Core teleoperation package
│   ├── cameras/                    # Camera handling and CameraManager
│   │   └── camera_manager.py
│   ├── detection/                  # Object detection (YOLOv8 wrapper)
│   │   └── object_detector.py
│   ├── gripper_pose/               # Gripper pose computation + utils
│   ├── hand_pose/                  # Hand pose factories and types
│   ├── kinematics/                 # URDF / FK / IK wrappers (pinocchio optional)
│   └── tracking/                   # Tracking & Kalman smoothing
├── scripts/                        # Utility scripts and wrappers (headless helpers)
├── src/                            # Installable package entry points (alternate layout)
├── tests/                          # Unit + integration tests (pytest)
├── yolov8n.pt                      # Example pretrained weights (optional)
├── main.py                         # Primary demo / entry script
├── pick_and_place.py               # Pick-and-place example harness
├── camera_setup.py                 # Camera helper utilities
├── test_gripper_only.py            # Quick smoke test script
├── environment.yml                 # Conda environment spec (recommended)
├── requirements.txt                # pip requirements (lighter alternative)
├── TROUBLESHOOTING.md              # Headless/runtime troubleshooting guide
├── CODE_INSPECTION_REPORT.md       # Recent code inspection summary
└── FIXES_APPLIED.md                # Summary of fixes applied in branch `jb_test`
```

## Installation

### Prerequisites (recommended)
- Python 3.8+ (3.10 used for development)
- Conda (recommended) or virtualenv
- Optional hardware: webcam/Arducam, SO-101 robot
- Optional GPU for fast YOLOv8 inference (CUDA-enabled)

Notes on optional dependencies:
- `pinocchio` (for advanced kinematics) is optional and only required if you use URDF/FK/IK features. The code lazy-loads `pinocchio` so core features work without it.
- YOLOv8 (`ultralytics`) is optional for object detection — you can use the system without it and add weights later.

### Recommended (Conda) install
1. Clone the repo and enter directory:
```bash
git clone https://github.com/ABMI-software/hand_controlLerobot.git
cd hand_controlLerobot
```

2. Create and activate the conda env:
```bash
conda env create -f environment.yml
conda activate hand_control
```

3. Install the package in editable mode:
```bash
pip install -e .
```

4. (Optional) Install `pinocchio` for kinematics (platform dependent):
Follow your platform's instructions; on many Linux systems a conda package is available.

### Pip / venv alternative
```bash
python -m venv hand_control
source hand_control/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Verify installation
Run a quick smoke test (headless-friendly):
```bash
# in base terminal (with env active)
python -c "import sys,cv2; print('Python', sys.version.split()[0]); print('OpenCV ok', cv2.__version__)"
pytest -q tests/test_utils.py
```

If you have a camera attached, try `python main.py` (see Usage). If no camera is attached, the system will fall back to a synthetic camera thread; you can also pass a video file path to `--cam-idx` to emulate a camera.

## Usage

### Basic Operation

#### Hand Tracking Only (local GUI)
Start the demo with a camera (index, device path, or video file):
```bash
# camera index 0
python main.py --cam-idx 0

# or use a video file for deterministic tests
python main.py --cam-idx /path/to/test_video.mp4
```

Notes:
- `--cam-idx` accepts either an integer camera index, a device path (e.g. `/dev/video0`) or a video file path.
- If no camera is available the system will fall back to a synthetic camera thread for CI/headless runs.

#### Robot Control / Teleoperation
Run teleoperation using the provided demo scripts. Example (adjust port and options for your hardware):
```bash
# example with SO-101 enabled (adjust serial port)
python3 scripts/gripper_direct_jog.py --hand right --model wilor --cam-idx 0 \
    --so101-enable --so101-port /dev/serial/by-id/usb-XXXXX --verbose
```

You can also run older demo wrappers (e.g. `test_gripper_only.py`) for quick smoke tests.

#### Object Detection and Picking
Start the pick-and-place harness (object detector optional):
```bash
python pick_and_place.py --cam-idx 0
```

Controls (keyboard while running):
- `a` : switch to auto-pick mode
- `t` : switch to teleoperation mode
- `q` : quit

Modes supported:
- Direct Control: joint-level control via hand gestures
- Task Space: end-effector Cartesian control
- Gripper Control: pinch gestures to open/close gripper

The pick-and-place harness will use YOLOv8 for detection if weights and `ultralytics` are available; otherwise it runs in degraded mode.

### Headless / CI-friendly Usage
For headless systems (CI, servers, WSL) set the environment variables to avoid GUI backends:
```bash
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
# then run headless script
bash scripts/run_poke_motor.sh
```

`scripts/run_poke_motor.sh` configures a headless environment and runs the demo; use it when no display server is available.

### Advanced Features
- Gesture Recording: Save and replay common movement sequences
- Safety Limits: Built-in joint and velocity limits
- Multiple Tracking Models: Switch between different hand tracking models

## Configuration

### Tracking Settings
```bash
# Select tracking model (options: mediapipe, wilor)
python main.py --tracker mediapipe

# Adjust tracking sensitivity
python main.py --sensitivity 0.8
```

### Robot & Hardware Settings
- Joint speed limits and robot-specific params: `config/robot_config.yaml` (create if absent)
- Gesture mappings: `config/gesture_mapping.yaml`
- Camera calibration: `config/camera_config.yaml` or use OpenCV calibration utilities

### Camera & Device Notes
- `--cam-idx` accepts an integer camera index, a device path (e.g. `/dev/video0`), or a video file path.
- When no camera is present the system uses a synthetic camera fallback so tests and demos remain runnable in CI. For automated tests prefer passing a short video file to `--cam-idx`.

### Optional Dependencies (summary)
- `pinocchio`: kinematics — optional, lazy-loaded
- `ultralytics` / YOLOv8: object detection — optional, used by `pick_and_place.py`
- `lerobot`: robot communication — required for SO-101 control

Add or pin these in your environment as needed; see `environment.yml` for the recommended development stack.

## Troubleshooting

### Common Issues
1. **Robot Not Detected**
   - Check USB connection
   - Verify correct port permissions
   - Run `python scan_bus.py` to detect connected devices

2. **Poor Tracking Performance**
   - Ensure good lighting conditions
   - Check webcam resolution settings
   - Try different tracking models

3. **Unexpected Robot Movement**
   - Verify calibration settings
   - Check gesture sensitivity settings
   - Ensure clean background for better tracking

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.