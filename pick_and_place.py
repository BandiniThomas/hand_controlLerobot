from typing import Optional, Tuple, Dict, List
import cv2
import numpy as np
from pathlib import Path
import time
import threading
from queue import Queue
import sys

from hand_teleop.cameras.camera_manager import CameraManager
from hand_teleop.detection.object_detector import ObjectDetector, DetectedObject
from hand_teleop.tracking.tracker import HandTracker
from hand_teleop.gripper_pose.gripper_pose import GripperPose

class PickingSystem:
    def __init__(
        self,
        arducam_id: int = 0,
        depth_camera_id: int = 1,  # Webcam or depth camera
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
    ):
        # Initialize camera manager
        self.camera_manager = CameraManager()
        try:
            self.camera_manager.add_camera("arducam", arducam_id, width=1280, height=720)
        except Exception as e:
            print(f"[warn] Arducam failed to initialize: {e}")
        
        try:
            self.camera_manager.add_camera("depth", depth_camera_id, width=640, height=480)
        except Exception as e:
            print(f"[warn] Depth camera failed to initialize: {e}")
        
        # Initialize object detector
        try:
            self.detector = ObjectDetector(model_path, conf_threshold)
        except Exception as e:
            print(f"[ERROR] Failed to load detector: {e}")
            self.detector = None
        
        # Initialize hand tracker (optional)
        try:
            self.hand_tracker = HandTracker(cam_idx=depth_camera_id, model="wilor")
        except Exception as e:
            print(f"[warn] Hand tracker failed to initialize: {e}")
            self.hand_tracker = None
        
        # Control flags
        self.running = False
        self.mode = "detection"  # "detection", "auto_pick", "teleop"
        self.detected_objects: List[DetectedObject] = []
        self.selected_object: Optional[DetectedObject] = None
        
        # User interface
        self.display_queue: Queue[Tuple[str, np.ndarray]] = Queue(maxsize=2)
        self._display_thread: Optional[threading.Thread] = None
        self._display_stop = threading.Event()

    def start(self):
        """Start the picking system."""
        self.running = True
        self._display_stop.clear()
        self.camera_manager.start()
        
        # Start display thread
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        
        print("\n" + "="*60)
        print("🤖 Picking System Started")
        print("="*60)
        print("Controls:")
        print("  'a' - Auto-pick mode (detects and picks nearest object)")
        print("  't' - Teleoperation mode (hand-controlled picking)")
        print("  'd' - Detection only mode (just shows detections)")
        print("  'q' or ESC - Quit")
        print("="*60 + "\n")
        
        # Main control loop
        while self.running:
            try:
                if self.mode == "detection":
                    self._run_detection()
                elif self.mode == "auto_pick":
                    self._run_auto_pick()
                elif self.mode == "teleop":
                    self._run_teleoperation()

                # Check for mode change commands (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # q or ESC
                    self.running = False
                elif key == ord('a'):
                    self.mode = "auto_pick"
                    print("[mode] Switched to Auto-Pick")
                elif key == ord('t'):
                    self.mode = "teleop"
                    print("[mode] Switched to Teleoperation")
                elif key == ord('d'):
                    self.mode = "detection"
                    print("[mode] Switched to Detection Only")
            except Exception as e:
                print(f"[ERROR] in main loop: {e}")
                break

    def _run_detection(self):
        """Run object detection on Arducam feed."""
        if not self.detector:
            return
        
        frame, _ = self.camera_manager.get_frame("arducam")
        if frame is not None:
            # Detect objects
            self.detected_objects = self.detector.detect(frame)
            
            # Visualize detections
            viz_frame = self.detector.visualize(frame.copy(), self.detected_objects)
            
            # Add instructions
            cv2.putText(viz_frame, "Detection Mode | Press 'a' for auto-pick, 't' for teleop, 'q' to quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Objects detected: {len(self.detected_objects)}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            try:
                self.display_queue.put_nowait(("Object Detection", viz_frame))
            except:
                pass  # Queue full, skip this frame

    def _run_auto_pick(self):
        """Run automatic picking mode (simplified version)."""
        if not self.detector:
            print("[ERROR] Detector not available for auto-pick")
            self.mode = "detection"
            return
        
        frame, _ = self.camera_manager.get_frame("arducam")
        if frame is None:
            return

        # Detect objects
        self.detected_objects = self.detector.detect(frame)
        if not self.detected_objects:
            print("[info] No objects detected. Switch to detection mode.")
            self.mode = "detection"
            return

        # Select nearest object if none selected
        if self.selected_object is None:
            self.selected_object = self.detected_objects[0]
            print(f"[pick] Selected object: {self.selected_object.label} (confidence: {self.selected_object.confidence:.2f})")

        # Visualize
        viz_frame = self.detector.visualize(frame.copy(), self.detected_objects)
        
        # Highlight selected object
        x1, y1, x2, y2 = map(int, self.selected_object.bbox)
        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow border for selected
        cv2.putText(viz_frame, f"Target: {self.selected_object.label}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(viz_frame, "Auto-Pick Mode | Computing path and gripper positioning...",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(viz_frame, f"[WIP] Path planning to {self.selected_object.label}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(viz_frame, "Steps: 1) Plan path  2) Position gripper  3) Grasp  4) Lift",
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        try:
            self.display_queue.put_nowait(("Auto-Pick Mode", viz_frame))
        except:
            pass

        # TODO: Implement actual picking sequence:
        # 1. Compute grasp point from object center/bbox
        # 2. Plan approach trajectory
        # 3. Move arm to approach pose
        # 4. Close gripper
        # 5. Move away
        # For now, just display the target

    def _run_teleoperation(self):
        """Run teleoperation mode (hand-controlled picking)."""
        if self.hand_tracker is None:
            print("[ERROR] Hand tracker not available for teleoperation")
            self.mode = "detection"
            return

        # Get frame from detection camera
        frame, _ = self.camera_manager.get_frame("arducam")
        if frame is not None:
            # Show detections
            if self.detector and self.detected_objects:
                frame = self.detector.visualize(frame.copy(), self.detected_objects)
            
            cv2.putText(frame, "Teleoperation Mode | Move hand to control gripper",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.putText(frame, "Hand pose tracking would control the arm...",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            try:
                self.display_queue.put_nowait(("Teleoperation Mode", frame))
            except:
                pass

            # TODO: Integrate with HandTracker for real-time control
            # pose = self.hand_tracker.read_hand_state(base_pose)
            # Convert pose to arm commands and send to robot

    def _display_loop(self):
        """Display thread for showing camera feeds and UI."""
        while not self._display_stop.is_set():
            try:
                # Use timeout to allow clean shutdown
                title, frame = self.display_queue.get(timeout=0.5)
                if frame is not None:
                    cv2.imshow(title, frame)
                    cv2.waitKey(1)
            except:
                # Timeout or queue empty
                pass

    def stop(self):
        """Stop the picking system."""
        print("\n[info] Stopping picking system...")
        self.running = False
        self._display_stop.set()
        if self._display_thread:
            self._display_thread.join(timeout=2.0)
        self.camera_manager.stop()
        cv2.destroyAllWindows()
        print("[done] Picking system stopped.\n")

def main():
    """Main entry point for picking system."""
    import argparse
    ap = argparse.ArgumentParser(description="Object picking system with detection and teleoperation.")
    ap.add_argument("--arducam-id", type=int, default=0, help="Arducam device index")
    ap.add_argument("--depth-id", type=int, default=1, help="Depth/hand tracking camera index")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    ap.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    
    args = ap.parse_args()

    # Initialize picking system
    try:
        system = PickingSystem(
            arducam_id=args.arducam_id,
            depth_camera_id=args.depth_id,
            model_path=args.model,
            conf_threshold=args.conf,
        )
        system.start()
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
    finally:
        system.stop()

if __name__ == "__main__":
    main()