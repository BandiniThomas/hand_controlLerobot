# Troubleshooting Guide

## Common Issues & Solutions

### 1. **"Failed to open camera at index/path: 0"**

**Problem:** The program crashes because no camera is detected.

**Solution:**
- **Check if a camera is connected:**
  ```bash
  ls /dev/video*
  ```
  If no output, you need to connect a camera (USB webcam or camera module).

- **Use a test video file instead:**
  ```bash
  python main.py --cam-idx /path/to/video.mp4
  python test_gripper_only.py --cam-idx /path/to/video.mp4
  ```

- **Test with a different camera index:**
  ```bash
  python main.py --cam-idx 1  # Try camera 1 instead of 0
  ```

---

### 2. **"ImportError: No module named 'pinocchio'"**

**Problem:** Kinematics module requires `pinocchio` which isn't installed.

**Solution:**
- Install pinocchio (conda only, not available on pip):
  ```bash
  conda install -c conda-forge pinocchio
  ```

- Note: If you're not using `--urdf-path` argument, pinocchio won't be loaded (lazy loading).

---

### 3. **"ModuleNotFoundError: No module named 'lerobot'"**

**Problem:** LeRobot robot control library isn't installed.

**Solution:**
```bash
conda activate hand_control
pip install lerobot==0.4.1
```

If that fails with dependency resolution issues:
```bash
pip install lerobot==0.4.1 --no-deps
# Then install missing dependencies individually as needed
```

---

### 4. **"qt.qpa.plugin: Could not find the Qt platform plugin 'wayland'"**

**Problem:** Qt/matplotlib GUI warnings when running on WSL or headless systems.

**Solution:**
- **For WSL:** Set environment variables before running:
  ```bash
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
  python main.py --cam-idx /path/to/video.mp4
  ```

- **Or use the provided wrapper script:**
  ```bash
  bash scripts/run_poke_motor.sh [args...]
  ```

---

### 5. **"Camera index out of range"**

**Problem:** OpenCV can't access the camera index you specified.

**Solution:**
- Try different indices (0, 1, 2, etc.):
  ```bash
  python main.py --cam-idx 1
  python main.py --cam-idx 2
  ```

- List available cameras:
  ```bash
  python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
  ```

---

### 6. **"Failed to connect to SO-101 robot"**

**Problem:** The robot arm connection fails.

**Solution:**
- Check USB connection:
  ```bash
  lsusb | grep -i "1a86"  # Look for the SO-101 device
  ```

- Verify correct serial port:
  ```bash
  ls /dev/serial/by-id/ | grep usb
  ```

- Update the port in the command:
  ```bash
  python main.py --so101-enable --so101-port /dev/ttyUSB0
  ```

- Run without robot to test hand tracking only:
  ```bash
  python main.py --cam-idx /path/to/video.mp4
  # Don't use --so101-enable flag
  ```

---

### 7. **"No hand detected" in hand tracking**

**Problem:** The vision module can't detect your hand.

**Solution:**
- Check lighting conditions (need good illumination)
- Try different hand pose estimator models:
  ```bash
  python main.py --cam-idx 0 --model mediapipe  # Instead of wilor
  python main.py --cam-idx 0 --model apriltag   # Requires AprilTag calibration
  ```

- Ensure your hand is within the camera frame
- Check that the hand is facing the camera

---

### 8. **"Failed to import lerobot"**

**Problem:** Dependency conflicts when using `pip install lerobot`.

**Solution:**
```bash
# Try installing with --no-deps first
pip install lerobot==0.4.1 --no-deps

# Then check which dependencies are missing
python -c "import lerobot; print('Success')"

# Install missing ones manually
pip install numpy opencv-python pydantic torch torchvision
```

---

### 9. **"MANO model, with only 10 shape coefficients"**

**Problem:** Warning message when hand tracking starts.

**Solution:**
- This is just a warning, not an error. The system works fine with this setting.
- Ignore the warning or suppress it:
  ```bash
  python -W ignore::FutureWarning main.py --cam-idx 0
  ```

---

### 10. **"Pose jump capped" warnings**

**Problem:** Hand tracking shows warnings about pose jumps being capped.

**Solution:**
- This is normal and indicates the Kalman filter is working (smoothing sudden jumps)
- If too frequent, check:
  - Hand lighting conditions
  - Camera is stable (not moving)
  - Hand tracking model is appropriate for your setup

---

## Environment Setup

### Fresh Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate hand_control

# 2. Install additional dependencies
pip install lerobot==0.4.1

# 3. Optional: Install kinematics support
conda install -c conda-forge pinocchio

# 4. Test basic setup
python test_gripper_only.py --cam-idx /path/to/test.mp4
```

### Verify Installation

```bash
# Test imports
python -c "from hand_teleop.tracking.tracker import HandTracker; print('✓ Tracker OK')"
python -c "from hand_teleop.detection.object_detector import ObjectDetector; print('✓ Detector OK')"
python -c "from hand_teleop.gripper_pose.gripper_pose_computer import GripperPoseComputer; print('✓ GripperPose OK')"
```

---

## Testing Without Hardware

### Using Test Video Files

```bash
# Any MP4, AVI, or MOV file works
python main.py --cam-idx recording.mp4 --hand right --model wilor
```

### Mock Mode (No Camera)

Currently, the system requires a camera source (real or video file). To develop without hardware:
1. Use pre-recorded video files
2. Create a mock camera class (contributor task)

---

## Performance Optimization

### If Running Slowly

```bash
# Use lower FPS
python main.py --cam-idx 0 --fps 15

# Use lighter hand tracking model (if available)
python main.py --cam-idx 0 --model mediapipe

# Reduce camera resolution
python test_gripper_only.py --cam-idx 0 --width 640 --height 480
```

### If Using Weak GPU/CPU

```bash
# Force CPU inference
python main.py --cam-idx 0 --device cpu
```

---

## Getting Help

### Enable Verbose Output

```bash
python main.py --cam-idx 0 --verbose
python test_gripper_only.py --cam-idx 0 --verbose
```

### Debug Mode

```bash
# Enables extra logging and checks
python main.py --cam-idx 0 --verbose 2>&1 | tee debug.log
```

### Report Issues

When reporting bugs, include:
1. Output of `conda env list` and `pip list`
2. Full error message with traceback
3. System info: `uname -a`
4. Camera setup details
5. Steps to reproduce

---

## See Also

- README.md - Project overview and usage
- CODE_INSPECTION_REPORT.md - Architecture and code structure
- environment.yml - Dependency specifications
