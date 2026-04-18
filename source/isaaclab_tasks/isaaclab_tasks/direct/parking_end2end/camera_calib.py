from __future__ import annotations

PARKING_CAMERA_KEYS: list[str] = ["camera_front", "camera_left", "camera_back", "camera_right"]

PARKING_CAMERA_MOUNTS: dict[str, dict[str, float | tuple[float, float, float]]] = {
    "camera_front": {"pos": (1.35, 0.0, 1.35), "yaw_deg": 0.0, "pitch_deg": -10.0},
    "camera_left": {"pos": (0.15, 0.95, 1.30), "yaw_deg": 90.0, "pitch_deg": -10.0},
    "camera_back": {"pos": (-1.35, 0.0, 1.30), "yaw_deg": 180.0, "pitch_deg": -10.0},
    "camera_right": {"pos": (0.15, -0.95, 1.30), "yaw_deg": -90.0, "pitch_deg": -10.0},
}

# Shared fisheye calibration used by both Isaac rendering and the FastRay LUT.
# Parameter order matches OpenCV fisheye / virtual_camera.FisheyeCamera.
PARKING_OPENCV_FISHEYE_CFG: dict[str, float] = {
    "cx": 317.25510702452,
    "cy": 190.15446946853075,
    "fx": 158.87579694498652,
    "fy": 158.87579694498652,
    "p0": 0.09929737309061659,
    "p1": 0.0,
    "p2": 0.0,
    "p3": 0.0,
}

# The LUT side also needs an explicit field-of-view gate for virtual_camera.FisheyeCamera.
PARKING_LOOKUP_FISHEYE_CFG: dict[str, float] = {
    **PARKING_OPENCV_FISHEYE_CFG,
    "fov": 200.0,
}
