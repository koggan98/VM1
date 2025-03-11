import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from realsense_depth import DepthCamera
from ur_commands_socket import URCommand  # Dein UR-Skript importieren
import time

# Keypoints nach COCO
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def get_model():
    model = YOLO('yolov8n-pose.pt')
    model.fuse()
    return model

def transform_to_robot_coords(x, y, z, calibration_matrix):
    # Transformation von Kamera- in Roboterkoordinaten
    cam_coords = np.array([x, y, z, 1])  # Homogene Koordinaten
    robot_coords = calibration_matrix @ cam_coords
    return robot_coords[:3]  # X, Y, Z für den Roboter zurückgeben

# Hauptteil
dc = DepthCamera()
model = get_model()

# UR3e Steuerung initialisieren
robot_ip = '192.168.1.11'
robot_command_port = 30002
robot_feedback_port = 30001
gripper_port = 63352
ur_robot = URCommand(robot_ip, robot_command_port, robot_feedback_port, gripper_port)

# Beispiel-Kalibrierungsmatrix
calibration_matrix = np.array([
    [1, 0, 0, 0.1],  # Beispielwerte, diese müssten durch Kalibrierung ermittelt werden
    [0, 1, 0, -0.05],
    [0, 0, 1, 0.2],
    [0, 0, 0, 1]
])

try:
    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        if not ret:
            continue

        results = model(color_frame)
        keypoints = results[0].keypoints.xy
        
        for person_keypoints in keypoints:
            for i, point in enumerate(person_keypoints):
                x, y = int(point[0]), int(point[1])
                
                if keypoint_names[i] == "Right Wrist":
                    z = depth_frame[y, x] if 0 <= y < depth_frame.shape[0] and 0 <= x < depth_frame.shape[1] else None
                    if z is not None:
                        robot_coords = transform_to_robot_coords(x, y, z, calibration_matrix)
                        print(f"Moving robot to {robot_coords}")
                        ur_robot.command_position(robot_coords, [0, -3.14, 0], speed=0.5)
                        time.sleep(1)
                    break  # Nur eine Person beachten
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    ur_robot.close_connections()
    dc.release()
    cv2.destroyAllWindows()
