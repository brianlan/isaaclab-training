#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Verify parking camera render/LUT alignment.")
parser.add_argument("--task", type=str, default="Isaac-Parking-End2End-Direct-v0")
parser.add_argument("--output-dir", type=str, default="/tmp/parking_camera_verify")
parser.add_argument("--skip-runtime-calibration", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import virtual_camera as vc

import isaaclab_tasks.direct.parking_end2end  # noqa: F401
from isaaclab_tasks.direct.parking_end2end import parking_env as parking_env_module
from isaaclab_tasks.direct.parking_end2end.agents.models import (
    IsaacCameraImage,
    IsaacCameraImageSet,
    VoxelLookUpTableGenerator,
    _build_camera_to_ego_rotation,
)
from isaaclab_tasks.direct.parking_end2end.camera_calib import (
    PARKING_CAMERA_KEYS,
    PARKING_CAMERA_MOUNTS,
    PARKING_LOOKUP_FISHEYE_CFG,
    PARKING_OPENCV_FISHEYE_CFG,
)
from isaaclab_tasks.direct.parking_end2end.parking_env import ParkingEnd2EndEnv, ParkingEnd2EndEnvCfg
from isaaclab_tasks.direct.parking_end2end.runtime_camera_api import get_runtime_camera_class


def _voxel_index_from_ego_point(
    point_xyz: np.ndarray,
    voxel_shape: tuple[int, int, int],
    voxel_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> int | None:
    z_count, x_count, y_count = voxel_shape
    fz = z_count / (voxel_range[0][1] - voxel_range[0][0])
    fx = x_count / (voxel_range[1][1] - voxel_range[1][0])
    fy = y_count / (voxel_range[2][1] - voxel_range[2][0])
    cz = -voxel_range[0][0] * fz - 0.5
    cx = -voxel_range[1][0] * fx - 0.5
    cy = -voxel_range[2][0] * fy - 0.5

    xx = int(np.round(point_xyz[0] * fx + cx))
    yy = int(np.round(point_xyz[1] * fy + cy))
    zz = int(np.round(point_xyz[2] * fz + cz))
    if zz < 0 or zz >= z_count or xx < 0 or xx >= x_count or yy < 0 or yy >= y_count:
        return None
    return zz * x_count * y_count + xx * y_count + yy


def _transform_local_points_to_ego(points_local: np.ndarray, vehicle_pose_local: np.ndarray) -> np.ndarray:
    rel = points_local.copy()
    rel[:, 0] -= vehicle_pose_local[0]
    rel[:, 1] -= vehicle_pose_local[1]
    yaw = vehicle_pose_local[2]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_ego = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1]
    y_ego = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
    return np.stack([x_ego, y_ego, rel[:, 2]], axis=-1)


def _make_virtual_camera(camera_key: str, resolution: tuple[int, int]) -> vc.FisheyeCamera:
    mount = PARKING_CAMERA_MOUNTS[camera_key]
    rotation = _build_camera_to_ego_rotation(
        float(mount["yaw_deg"]),
        float(mount["pitch_deg"]),
    ).cpu().numpy().astype(np.float32)
    translation = np.asarray(mount["pos"], dtype=np.float32)
    intrinsic = np.array(
        [
            PARKING_LOOKUP_FISHEYE_CFG["cx"],
            PARKING_LOOKUP_FISHEYE_CFG["cy"],
            PARKING_LOOKUP_FISHEYE_CFG["fx"],
            PARKING_LOOKUP_FISHEYE_CFG["fy"],
            PARKING_LOOKUP_FISHEYE_CFG["p0"],
            PARKING_LOOKUP_FISHEYE_CFG["p1"],
            PARKING_LOOKUP_FISHEYE_CFG["p2"],
            PARKING_LOOKUP_FISHEYE_CFG["p3"],
        ],
        dtype=np.float32,
    )
    return vc.FisheyeCamera(
        resolution=resolution,
        extrinsic=(rotation, translation),
        intrinsic=intrinsic,
        fov=float(PARKING_LOOKUP_FISHEYE_CFG["fov"]),
    )


def _obstacle_corners(obstacle) -> np.ndarray:
    hx = 0.5 * obstacle.size_xy[0]
    hy = 0.5 * obstacle.size_xy[1]
    hz = 0.5 * max(obstacle.height, 0.1)
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    cos_yaw = math.cos(obstacle.yaw)
    sin_yaw = math.sin(obstacle.yaw)
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center = np.array([obstacle.center_xy[0], obstacle.center_xy[1], obstacle.z], dtype=np.float32)
    return local @ rot.T + center


def _draw_polyline(image: np.ndarray, points_uv: np.ndarray, color: tuple[int, int, int], closed: bool = True) -> None:
    valid = np.all(points_uv >= 0, axis=1)
    if valid.sum() < 2:
        return
    pts = points_uv[valid].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "camera_runtime": {},
        "lut_vs_direct_error": {},
    }
    Camera = get_runtime_camera_class()

    if args_cli.skip_runtime_calibration:
        def _skip_runtime_calibration(self) -> None:
            return None

        ParkingEnd2EndEnv._apply_render_camera_calibration = _skip_runtime_calibration
        parking_env_module.ParkingEnd2EndEnv._apply_render_camera_calibration = _skip_runtime_calibration
        report["runtime_calibration"] = {"status": "skipped"}

    cfg = ParkingEnd2EndEnvCfg()
    cfg.scene.num_envs = 1
    env = ParkingEnd2EndEnv(cfg=cfg)
    obs, _ = env.reset()
    env_unwrapped = env

    runtime_cameras = {}
    env_path = env_unwrapped.scene.env_prim_paths[0]
    for camera_key in PARKING_CAMERA_KEYS:
        prim_path = f"{env_path}/Vehicle/{camera_key}_sensor"
        camera = Camera(prim_path=prim_path, resolution=(cfg.camera_width, cfg.camera_height))
        runtime_cameras[camera_key] = camera
        model = camera.get_lens_distortion_model()
        props = camera.get_opencv_fisheye_properties()
        if props is None:
            cx = cy = fx = fy = None
            coeffs = None
        else:
            cx, cy, fx, fy, coeffs = props
        report["camera_runtime"][camera_key] = {
            "prim_path": prim_path,
            "lens_model": model,
            "cx": cx,
            "cy": cy,
            "fx": fx,
            "fy": fy,
            "coeffs": list(coeffs) if coeffs is not None else None,
        }

    voxel_shape = (8, 160, 120)
    voxel_range = ((-1.0, 3.0), (12.0, -12.0), (9.0, -9.0))
    blank_rgb = np.zeros((cfg.camera_height, cfg.camera_width, 3), dtype=np.uint8)
    blank_mask = np.ones((cfg.camera_height, cfg.camera_width), dtype=np.uint8)
    camera_images = {}
    for camera_key in PARKING_CAMERA_KEYS:
        vcam = _make_virtual_camera(camera_key, (cfg.camera_width, cfg.camera_height))
        camera_images[camera_key] = IsaacCameraImage(
            cam_id=camera_key,
            cam_type="FisheyeCamera",
            img=blank_rgb,
            ego_mask=blank_mask,
            extrinsic=vcam.extrinsic,
            intrinsic=np.array(vcam.intrinsic, dtype=np.float32),
        )
    lut_gen = VoxelLookUpTableGenerator(
        voxel_feature_config={
            "voxel_shape": voxel_shape,
            "voxel_range": voxel_range,
            "ego_distance_max": 16.0,
            "ego_distance_step": 2.0,
        },
        camera_feature_configs={
            key: {
                "ray_distance_num_channel": 32,
                "ray_distance_start": 0.5,
                "ray_distance_step": 0.5,
                "feature_downscale": 1,
            }
            for key in PARKING_CAMERA_KEYS
        },
        bilinear_interpolation=False,
    )
    lut = lut_gen.generate(IsaacCameraImageSet(camera_images))

    scene_index = int(env_unwrapped.scene_ids[0].item())
    scene_info = env_unwrapped.scene_infos[scene_index]
    vehicle_pose_local = env_unwrapped.vehicle_pose_local[0].detach().cpu().numpy()

    scene_points_local: list[np.ndarray] = []
    slot_polygons_local: list[np.ndarray] = []
    obstacle_polygons_local: list[np.ndarray] = []

    for slot in scene_info.slots:
        verts = np.array([[x, y, 0.0] for x, y in slot.vertices_xy], dtype=np.float32)
        slot_polygons_local.append(verts)
        scene_points_local.extend([p.copy() for p in verts])

    for obstacle in scene_info.obstacles:
        corners = _obstacle_corners(obstacle)
        obstacle_polygons_local.append(corners)
        scene_points_local.extend([p.copy() for p in corners])

    scene_points_local_np = np.array(scene_points_local, dtype=np.float32)
    scene_points_ego = _transform_local_points_to_ego(scene_points_local_np, vehicle_pose_local)

    for camera_key in PARKING_CAMERA_KEYS:
        image = (obs[camera_key][0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        vcam = _make_virtual_camera(camera_key, (cfg.camera_width, cfg.camera_height))
        uu, vv = vcam.project_points_from_camera_to_image(scene_points_ego.T)
        direct_uv = np.stack([uu, vv], axis=-1)

        lut_uv = np.full_like(direct_uv, -1.0)
        point_errors: list[float] = []
        for idx, point_ego in enumerate(scene_points_ego):
            voxel_idx = _voxel_index_from_ego_point(point_ego, voxel_shape, voxel_range)
            if voxel_idx is None:
                continue
            uu_lut = lut[camera_key]["uu"][voxel_idx]
            vv_lut = lut[camera_key]["vv"][voxel_idx]
            if uu_lut < 0 or vv_lut < 0 or direct_uv[idx, 0] < 0 or direct_uv[idx, 1] < 0:
                continue
            lut_uv[idx] = np.array([uu_lut, vv_lut], dtype=np.float32)
            point_errors.append(float(np.linalg.norm(direct_uv[idx] - lut_uv[idx])))

        report["lut_vs_direct_error"][camera_key] = {
            "num_points_compared": len(point_errors),
            "mean_pixel_error": float(np.mean(point_errors)) if point_errors else None,
            "max_pixel_error": float(np.max(point_errors)) if point_errors else None,
        }

        point_offset = 0
        for slot_poly in slot_polygons_local:
            count = len(slot_poly)
            points_uv = direct_uv[point_offset : point_offset + count]
            _draw_polyline(image_bgr, points_uv, (0, 255, 0), closed=True)
            for pt in points_uv:
                if pt[0] >= 0 and pt[1] >= 0:
                    cv2.circle(image_bgr, tuple(np.round(pt).astype(int)), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            point_offset += count

        for obstacle_poly in obstacle_polygons_local:
            count = len(obstacle_poly)
            points_uv = direct_uv[point_offset : point_offset + count]
            base = points_uv[:4]
            top = points_uv[4:]
            _draw_polyline(image_bgr, base, (255, 255, 0), closed=True)
            _draw_polyline(image_bgr, top, (255, 255, 0), closed=True)
            for i in range(4):
                if np.all(base[i] >= 0) and np.all(top[i] >= 0):
                    cv2.line(
                        image_bgr,
                        tuple(np.round(base[i]).astype(int)),
                        tuple(np.round(top[i]).astype(int)),
                        (255, 255, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )
            point_offset += count

        valid_lut = np.all(lut_uv >= 0, axis=1)
        for pt in lut_uv[valid_lut]:
            cv2.circle(image_bgr, tuple(np.round(pt).astype(int)), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        cv2.putText(
            image_bgr,
            f"{camera_key} direct=green lut=red box=cyan",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"{camera_key}_overlay.png"), image_bgr)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
