from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import FisheyeCameraCfg, SimulationCfg
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .camera_calib import PARKING_CAMERA_KEYS, PARKING_CAMERA_MOUNTS, PARKING_OPENCV_FISHEYE_CFG
from .runtime_camera_api import get_runtime_camera_class
from .scene_utils import SceneInfo, discover_scenes


_VEHICLE_BASE_ORIENTATION = (
    float(quat_from_euler_xyz(torch.tensor(math.pi * 0.5), torch.tensor(0.0), torch.tensor(math.pi * 0.5))[0]),
    float(quat_from_euler_xyz(torch.tensor(math.pi * 0.5), torch.tensor(0.0), torch.tensor(math.pi * 0.5))[1]),
    float(quat_from_euler_xyz(torch.tensor(math.pi * 0.5), torch.tensor(0.0), torch.tensor(math.pi * 0.5))[2]),
    float(quat_from_euler_xyz(torch.tensor(math.pi * 0.5), torch.tensor(0.0), torch.tensor(math.pi * 0.5))[3]),
)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rotation_matrix(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ],
        dim=-2,
    )


def _obb_overlap_2d(
    ego_center: torch.Tensor,
    ego_yaw: torch.Tensor,
    ego_half_extents: torch.Tensor,
    obs_center: torch.Tensor,
    obs_yaw: torch.Tensor,
    obs_half_extents: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    ego_axes = _rotation_matrix(ego_yaw)
    obs_axes = _rotation_matrix(obs_yaw)
    t = obs_center - ego_center[:, None, :]

    axes_a0 = ego_axes[:, 0, :]
    axes_a1 = ego_axes[:, 1, :]
    axes_b0 = obs_axes[:, :, 0, :]
    axes_b1 = obs_axes[:, :, 1, :]

    t_a0 = torch.sum(t * axes_a0[:, None, :], dim=-1).abs()
    t_a1 = torch.sum(t * axes_a1[:, None, :], dim=-1).abs()
    r00 = torch.sum(axes_a0[:, None, :] * axes_b0, dim=-1).abs() + 1.0e-6
    r01 = torch.sum(axes_a0[:, None, :] * axes_b1, dim=-1).abs() + 1.0e-6
    r10 = torch.sum(axes_a1[:, None, :] * axes_b0, dim=-1).abs() + 1.0e-6
    r11 = torch.sum(axes_a1[:, None, :] * axes_b1, dim=-1).abs() + 1.0e-6
    a0 = ego_half_extents[0]
    a1 = ego_half_extents[1]
    b0 = obs_half_extents[..., 0]
    b1 = obs_half_extents[..., 1]

    overlap_a0 = t_a0 <= a0 + b0 * r00 + b1 * r01
    overlap_a1 = t_a1 <= a1 + b0 * r10 + b1 * r11

    t_b0 = torch.sum(t * axes_b0, dim=-1).abs()
    t_b1 = torch.sum(t * axes_b1, dim=-1).abs()
    overlap_b0 = t_b0 <= b0 + a0 * r00 + a1 * r10
    overlap_b1 = t_b1 <= b1 + a0 * r01 + a1 * r11
    overlap = overlap_a0 & overlap_a1 & overlap_b0 & overlap_b1 & obs_mask
    return torch.any(overlap, dim=1)


def _point_in_slot(point_xy: torch.Tensor, slot_center_xy: torch.Tensor, slot_yaw: torch.Tensor, slot_half_extents: torch.Tensor) -> torch.Tensor:
    rel = point_xy - slot_center_xy
    rot = _rotation_matrix(-slot_yaw)
    local = torch.einsum("bij,bj->bi", rot, rel)
    return (local[:, 0].abs() <= slot_half_extents[:, 0]) & (local[:, 1].abs() <= slot_half_extents[:, 1])


@configclass
class ParkingEnd2EndEnvCfg(DirectRLEnvCfg):
    episode_length_s = 30.0
    decimation = 10
    is_finite_horizon = False
    rerender_on_reset = False
    num_rerenders_on_reset = 1

    scene_dataset_root = "/ssd5/datasets/mv-parking-3dgs/scenes"
    generated_scene_asset_root = "/tmp/isaaclab_parking_scene_assets"
    ego_vehicle_usdz_path = "/home/rlan/projects/vla-robot-demo/2025_BMW_M5_Sedan.usdz"

    sim: SimulationCfg = SimulationCfg(dt=0.05, render_interval=10)
    viewer = ViewerCfg(eye=(15.0, 15.0, 15.0))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=80.0, replicate_physics=False, clone_in_fabric=False)

    camera_height = 384
    camera_width = 640
    action_space = 20
    observation_space = gym.spaces.Dict(
        {
            "camera_front": gym.spaces.Box(low=0.0, high=1.0, shape=(3, camera_height, camera_width), dtype=float),
            "camera_left": gym.spaces.Box(low=0.0, high=1.0, shape=(3, camera_height, camera_width), dtype=float),
            "camera_back": gym.spaces.Box(low=0.0, high=1.0, shape=(3, camera_height, camera_width), dtype=float),
            "camera_right": gym.spaces.Box(low=0.0, high=1.0, shape=(3, camera_height, camera_width), dtype=float),
            "goal": gym.spaces.Box(low=-1000.0, high=1000.0, shape=(4,), dtype=float),
            "kinematics": gym.spaces.Box(low=-1000.0, high=1000.0, shape=(6,), dtype=float),
        }
    )
    state_space = 0

    vehicle_length = 5.10
    vehicle_width = 1.97
    wheelbase = 3.00
    max_steer_angle_rad = 0.60
    max_speed_mps = 2.0
    max_yaw_rate_radps = 1.2
    vehicle_z = 0.3

    success_pos_thresh_m = 0.5
    success_heading_thresh_deg = 15.0
    low_speed_thresh_mps = 0.25
    obstacle_clearance_margin_m = 0.75

    reward_progress_weight = 2.5
    reward_success = 25.0
    reward_collision = -25.0
    reward_timeout = -2.0
    reward_action_l2_weight = -0.02
    reward_action_smoothness_weight = -0.03
    reward_clearance_weight = -0.8
    reward_near_goal_stop_weight = -0.3
    reward_alive = 0.05


class ParkingEnd2EndEnv(DirectRLEnv):
    cfg: ParkingEnd2EndEnvCfg

    def __init__(self, cfg: ParkingEnd2EndEnvCfg, render_mode: str | None = None, **kwargs):
        self.scene_infos: list[SceneInfo] = discover_scenes(cfg.scene_dataset_root, cfg.generated_scene_asset_root)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.scene_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % len(self.scene_infos)
        self.scene_local_bounds = torch.tensor(
            [[[s.bounds_xy[0][0], s.bounds_xy[0][1]], [s.bounds_xy[1][0], s.bounds_xy[1][1]]] for s in self.scene_infos],
            device=self.device,
            dtype=torch.float32,
        )

        self.max_obstacles = max(len(s.obstacles) for s in self.scene_infos)
        self.max_slots = max(len(s.slots) for s in self.scene_infos)
        self.obstacle_centers = torch.zeros(len(self.scene_infos), self.max_obstacles, 2, device=self.device)
        self.obstacle_half_extents = torch.zeros(len(self.scene_infos), self.max_obstacles, 2, device=self.device)
        self.obstacle_yaws = torch.zeros(len(self.scene_infos), self.max_obstacles, device=self.device)
        self.obstacle_mask = torch.zeros(len(self.scene_infos), self.max_obstacles, device=self.device, dtype=torch.bool)
        self.slot_centers = torch.zeros(len(self.scene_infos), self.max_slots, 2, device=self.device)
        self.slot_yaws = torch.zeros(len(self.scene_infos), self.max_slots, device=self.device)
        self.slot_half_extents = torch.zeros(len(self.scene_infos), self.max_slots, 2, device=self.device)
        self.slot_parkable = torch.zeros(len(self.scene_infos), self.max_slots, device=self.device, dtype=torch.bool)
        self.slot_mask = torch.zeros(len(self.scene_infos), self.max_slots, device=self.device, dtype=torch.bool)
        for scene_index, scene_info in enumerate(self.scene_infos):
            for obstacle_index, obstacle in enumerate(scene_info.obstacles):
                self.obstacle_centers[scene_index, obstacle_index] = torch.tensor(obstacle.center_xy, device=self.device)
                self.obstacle_half_extents[scene_index, obstacle_index] = 0.5 * torch.tensor(obstacle.size_xy, device=self.device)
                self.obstacle_yaws[scene_index, obstacle_index] = obstacle.yaw
                self.obstacle_mask[scene_index, obstacle_index] = True
            for slot_index, slot in enumerate(scene_info.slots):
                length = 0.5 * math.dist(slot.vertices_xy[0], slot.vertices_xy[1]) + 0.5 * math.dist(slot.vertices_xy[2], slot.vertices_xy[3])
                width = 0.5 * math.dist(slot.vertices_xy[1], slot.vertices_xy[2]) + 0.5 * math.dist(slot.vertices_xy[3], slot.vertices_xy[0])
                self.slot_centers[scene_index, slot_index] = torch.tensor(slot.center_xy, device=self.device)
                self.slot_yaws[scene_index, slot_index] = slot.yaw
                self.slot_half_extents[scene_index, slot_index] = torch.tensor([0.5 * length, 0.5 * width], device=self.device)
                self.slot_parkable[scene_index, slot_index] = slot.parkable
                self.slot_mask[scene_index, slot_index] = True

        self.vehicle_pose_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.vehicle_state = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_pose_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_goal_pose_body = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_goal_obs = torch.zeros(self.num_envs, 4, device=self.device)
        self.kinematics_obs = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_action_seq = torch.zeros(self.num_envs, 10, 2, device=self.device)
        self.processed_action_seq = torch.zeros_like(self.last_action_seq)
        self.substep_index = 0
        self._reward_step = torch.zeros(self.num_envs, device=self.device)
        self._collision_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._success_flag = torch.zeros_like(self._collision_flag)
        self._done_during_sequence = torch.zeros_like(self._collision_flag)
        self.vehicle_half_extents = torch.tensor(
            [0.5 * self.cfg.vehicle_length + 0.1, 0.5 * self.cfg.vehicle_width + 0.1], device=self.device
        )

    def _setup_scene(self):
        for env_path in self.scene.env_prim_paths:
            vehicle_spawn_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.ego_vehicle_usdz_path)
            vehicle_spawn_cfg.func(f"{env_path}/Vehicle", vehicle_spawn_cfg)
            sim_utils.standardize_xform_ops(
                self.scene.stage.GetPrimAtPath(f"{env_path}/Vehicle"),
                translation=(0.0, 0.0, self.cfg.vehicle_z),
                orientation=_VEHICLE_BASE_ORIENTATION,
                scale=(1.0, 1.0, 1.0),
            )
        self.vehicle_view = XformPrimView(
            "/World/envs/env_.*/Vehicle",
            device=self.device,
            validate_xform_ops=False,
            sync_usd_on_fabric_write=True,
            stage=self.scene.stage,
        )
        self.scene.extras["vehicle"] = self.vehicle_view

        def camera_cfg(name: str, pos: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> TiledCameraCfg:
            euler = torch.tensor([math.radians(v) for v in euler_deg], dtype=torch.float32)
            quat = quat_from_euler_xyz(euler[0], euler[1], euler[2])
            return TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/Vehicle/{name}",
                update_period=0.0,
                height=self.cfg.camera_height,
                width=self.cfg.camera_width,
                data_types=["rgb"],
                spawn=FisheyeCameraCfg(
                    projection_type="fisheyePolynomial",
                    fisheye_max_fov=200.0,
                    fisheye_nominal_width=float(self.cfg.camera_width),
                    fisheye_nominal_height=float(self.cfg.camera_height),
                    fisheye_optical_centre_x=float(self.cfg.camera_width) * 0.5,
                    fisheye_optical_centre_y=float(self.cfg.camera_height) * 0.5,
                    clipping_range=(0.05, 60.0),
                ),
                offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=tuple(float(v) for v in quat), convention="world"),
            )

        self.camera_front = TiledCamera(
            camera_cfg(
                "camera_front_sensor",
                tuple(PARKING_CAMERA_MOUNTS["camera_front"]["pos"]),
                (float(PARKING_CAMERA_MOUNTS["camera_front"]["pitch_deg"]), 0.0, float(PARKING_CAMERA_MOUNTS["camera_front"]["yaw_deg"])),
            )
        )
        self.camera_left = TiledCamera(
            camera_cfg(
                "camera_left_sensor",
                tuple(PARKING_CAMERA_MOUNTS["camera_left"]["pos"]),
                (float(PARKING_CAMERA_MOUNTS["camera_left"]["pitch_deg"]), 0.0, float(PARKING_CAMERA_MOUNTS["camera_left"]["yaw_deg"])),
            )
        )
        self.camera_back = TiledCamera(
            camera_cfg(
                "camera_back_sensor",
                tuple(PARKING_CAMERA_MOUNTS["camera_back"]["pos"]),
                (float(PARKING_CAMERA_MOUNTS["camera_back"]["pitch_deg"]), 0.0, float(PARKING_CAMERA_MOUNTS["camera_back"]["yaw_deg"])),
            )
        )
        self.camera_right = TiledCamera(
            camera_cfg(
                "camera_right_sensor",
                tuple(PARKING_CAMERA_MOUNTS["camera_right"]["pos"]),
                (float(PARKING_CAMERA_MOUNTS["camera_right"]["pitch_deg"]), 0.0, float(PARKING_CAMERA_MOUNTS["camera_right"]["yaw_deg"])),
            )
        )
        self.scene.sensors["camera_front"] = self.camera_front
        self.scene.sensors["camera_left"] = self.camera_left
        self.scene.sensors["camera_back"] = self.camera_back
        self.scene.sensors["camera_right"] = self.camera_right

        self._apply_render_camera_calibration()

        for env_index, env_path in enumerate(self.scene.env_prim_paths):
            scene_usd = self.scene_infos[env_index % len(self.scene_infos)].composite_usda_path
            scene_spawn_cfg = sim_utils.UsdFileCfg(usd_path=str(scene_usd), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
            scene_spawn_cfg.func(f"{env_path}/ParkingScene", scene_spawn_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_render_camera_calibration(self) -> None:
        Camera = get_runtime_camera_class()

        distortion = [
            float(PARKING_OPENCV_FISHEYE_CFG["p0"]),
            float(PARKING_OPENCV_FISHEYE_CFG["p1"]),
            float(PARKING_OPENCV_FISHEYE_CFG["p2"]),
            float(PARKING_OPENCV_FISHEYE_CFG["p3"]),
        ]
        for env_path in self.scene.env_prim_paths:
            for camera_key in PARKING_CAMERA_KEYS:
                prim_path = f"{env_path}/Vehicle/{camera_key}_sensor"
                camera = Camera(
                    prim_path=prim_path,
                    resolution=(self.cfg.camera_width, self.cfg.camera_height),
                )
                camera.set_opencv_fisheye_properties(
                    cx=float(PARKING_OPENCV_FISHEYE_CFG["cx"]),
                    cy=float(PARKING_OPENCV_FISHEYE_CFG["cy"]),
                    fx=float(PARKING_OPENCV_FISHEYE_CFG["fx"]),
                    fy=float(PARKING_OPENCV_FISHEYE_CFG["fy"]),
                    fisheye=distortion,
                )

    def _transform_goal_to_body(self, goal_xy: torch.Tensor, goal_yaw: torch.Tensor, vehicle_xy: torch.Tensor, vehicle_yaw: torch.Tensor) -> torch.Tensor:
        rel = goal_xy - vehicle_xy
        cos_yaw = torch.cos(vehicle_yaw)
        sin_yaw = torch.sin(vehicle_yaw)
        x_body = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1]
        y_body = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
        yaw_body = _wrap_to_pi(goal_yaw - vehicle_yaw)
        return torch.stack([x_body, y_body, torch.sin(yaw_body), torch.cos(yaw_body)], dim=-1)

    def _compute_pose_potential(self, goal_obs: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.norm(goal_obs[:, :2], dim=-1)
        heading = torch.atan2(goal_obs[:, 2], goal_obs[:, 3]).abs()
        near_goal = distance < 1.5
        return distance + 0.5 * heading + 0.4 * near_goal.float() * speed

    def _sample_curriculum(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        progress = min(float(self.common_step_counter) / 200000.0, 1.0)
        longitudinal = (-4.0 - 4.0 * progress, 1.0 + 3.0 * progress)
        lateral = (-1.0 - 5.0 * progress, 1.0 + 5.0 * progress)
        yaw_noise = math.radians(15.0 + 165.0 * progress)
        return longitudinal, lateral, yaw_noise

    def _sample_spawn_pose(self, scene_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        longitudinal_range, lateral_range, yaw_noise = self._sample_curriculum()
        slot_mask = self.slot_mask[scene_id]
        slot_indices = torch.where(slot_mask & self.slot_parkable[scene_id])[0]
        if len(slot_indices) == 0:
            slot_indices = torch.where(slot_mask)[0]
        goal_slot_index = slot_indices[torch.randint(len(slot_indices), (1,), device=self.device)[0]]

        goal_pose = torch.stack(
            [
                self.slot_centers[scene_id, goal_slot_index, 0],
                self.slot_centers[scene_id, goal_slot_index, 1],
                self.slot_yaws[scene_id, goal_slot_index],
            ]
        )
        goal_rot = _rotation_matrix(goal_pose[2].view(1))[0]
        bounds = self.scene_local_bounds[scene_id]

        for _ in range(64):
            local = torch.tensor(
                [
                    torch.empty((), device=self.device).uniform_(*longitudinal_range),
                    torch.empty((), device=self.device).uniform_(*lateral_range),
                ],
                device=self.device,
            )
            spawn_xy = goal_pose[:2] + goal_rot @ local
            spawn_yaw = goal_pose[2] + torch.empty((), device=self.device).uniform_(-yaw_noise, yaw_noise)
            in_bounds = bool(
                (spawn_xy[0] >= bounds[0, 0])
                & (spawn_xy[0] <= bounds[0, 1])
                & (spawn_xy[1] >= bounds[1, 0])
                & (spawn_xy[1] <= bounds[1, 1])
            )
            if not in_bounds:
                continue
            inside_slot = _point_in_slot(
                spawn_xy.view(1, 2),
                self.slot_centers[scene_id, slot_mask],
                self.slot_yaws[scene_id, slot_mask],
                self.slot_half_extents[scene_id, slot_mask] + 0.5,
            ).any()
            colliding = _obb_overlap_2d(
                spawn_xy.view(1, 2),
                spawn_yaw.view(1),
                self.vehicle_half_extents,
                self.obstacle_centers[scene_id].view(1, self.max_obstacles, 2),
                self.obstacle_yaws[scene_id].view(1, self.max_obstacles),
                self.obstacle_half_extents[scene_id].view(1, self.max_obstacles, 2),
                self.obstacle_mask[scene_id].view(1, self.max_obstacles),
            )[0]
            if not bool(inside_slot) and not bool(colliding):
                return goal_pose, torch.stack([spawn_xy[0], spawn_xy[1], spawn_yaw])
        return goal_pose, goal_pose + torch.tensor([-4.0, 0.0, 0.0], device=self.device)

    def _write_vehicle_pose_to_sim(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        yaws = self.vehicle_pose_local[env_ids, 2]
        yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaws), torch.zeros_like(yaws), yaws)
        base_quat = torch.tensor(_VEHICLE_BASE_ORIENTATION, device=self.device).expand(len(env_ids), -1)
        quat = quat_mul(yaw_quat, base_quat)
        positions = torch.zeros(len(env_ids), 3, device=self.device)
        positions[:, 0:2] = self.vehicle_pose_local[env_ids, :2] + self.scene.env_origins[env_ids, :2]
        positions[:, 2] = self.cfg.vehicle_z
        self.vehicle_view.set_world_poses(positions=positions, orientations=quat, indices=env_ids.tolist())

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, 10, 2).clamp(-1.0, 1.0)
        self.processed_action_seq[..., 0] = actions[..., 0] * self.cfg.max_speed_mps
        self.processed_action_seq[..., 1] = actions[..., 1] * self.cfg.max_yaw_rate_radps
        self.last_action_seq.copy_(self.processed_action_seq)
        self.substep_index = 0
        self._reward_step.zero_()
        self._collision_flag.zero_()
        self._success_flag.zero_()
        self._done_during_sequence.zero_()
        self.prev_goal_pose_body.copy_(self._transform_goal_to_body(
            self.goal_pose_local[:, :2],
            self.goal_pose_local[:, 2],
            self.vehicle_pose_local[:, :2],
            self.vehicle_pose_local[:, 2],
        ))

    def _apply_action(self) -> None:
        cmd = self.processed_action_seq[:, self.substep_index]
        dt = self.physics_dt
        active = ~self._done_during_sequence
        if active.any():
            env_ids = torch.where(active)[0]
            v_cmd = cmd[env_ids, 0]
            omega_cmd = cmd[env_ids, 1]
            max_omega = torch.abs(v_cmd) * math.tan(self.cfg.max_steer_angle_rad) / max(self.cfg.wheelbase, 1.0e-3)
            omega = torch.clamp(omega_cmd, -torch.maximum(max_omega, torch.full_like(max_omega, 0.15)), torch.maximum(max_omega, torch.full_like(max_omega, 0.15)))
            yaw = self.vehicle_pose_local[env_ids, 2]
            self.vehicle_pose_local[env_ids, 0] += v_cmd * torch.cos(yaw) * dt
            self.vehicle_pose_local[env_ids, 1] += v_cmd * torch.sin(yaw) * dt
            self.vehicle_pose_local[env_ids, 2] = _wrap_to_pi(yaw + omega * dt)
            self.vehicle_state[env_ids, 0] = v_cmd
            self.vehicle_state[env_ids, 1] = omega
            self._write_vehicle_pose_to_sim(env_ids)

            goal_obs = self._transform_goal_to_body(
                self.goal_pose_local[env_ids, :2],
                self.goal_pose_local[env_ids, 2],
                self.vehicle_pose_local[env_ids, :2],
                self.vehicle_pose_local[env_ids, 2],
            )
            prev_goal = self.prev_goal_pose_body[env_ids]
            progress = self._compute_pose_potential(prev_goal, self.vehicle_state[env_ids, 0].abs()) - self._compute_pose_potential(goal_obs, self.vehicle_state[env_ids, 0].abs())
            self._reward_step[env_ids] += self.cfg.reward_progress_weight * progress

            scene_ids = self.scene_ids[env_ids]
            colliding = _obb_overlap_2d(
                self.vehicle_pose_local[env_ids, :2],
                self.vehicle_pose_local[env_ids, 2],
                self.vehicle_half_extents,
                self.obstacle_centers[scene_ids],
                self.obstacle_yaws[scene_ids],
                self.obstacle_half_extents[scene_ids],
                self.obstacle_mask[scene_ids],
            )
            self._collision_flag[env_ids] |= colliding

            clearance_radius = torch.linalg.norm(self.vehicle_half_extents)
            obstacle_radius = torch.linalg.norm(self.obstacle_half_extents[scene_ids], dim=-1)
            center_dist = torch.linalg.norm(self.obstacle_centers[scene_ids] - self.vehicle_pose_local[env_ids, None, :2], dim=-1)
            clearance = center_dist - (clearance_radius + obstacle_radius)
            clearance = torch.where(self.obstacle_mask[scene_ids], clearance, torch.full_like(clearance, 1.0e6))
            min_clearance = torch.min(clearance, dim=1).values
            clearance_penalty = torch.clamp(self.cfg.obstacle_clearance_margin_m - min_clearance, min=0.0) / self.cfg.obstacle_clearance_margin_m
            self._reward_step[env_ids] += self.cfg.reward_clearance_weight * clearance_penalty

            distance = torch.linalg.norm(goal_obs[:, :2], dim=-1)
            heading_error = torch.atan2(goal_obs[:, 2], goal_obs[:, 3]).abs()
            success = (distance <= self.cfg.success_pos_thresh_m) & (heading_error <= math.radians(self.cfg.success_heading_thresh_deg))
            self._success_flag[env_ids] |= success
            self._done_during_sequence[env_ids] |= colliding | success
            self.prev_goal_pose_body[env_ids] = goal_obs

        self.substep_index += 1

    def _get_observations(self) -> dict:
        def _camera(sensor: TiledCamera) -> torch.Tensor:
            rgb = sensor.data.output["rgb"][..., :3].float() / 255.0
            return rgb.permute(0, 3, 1, 2).contiguous()

        self.current_goal_obs = self._transform_goal_to_body(
            self.goal_pose_local[:, :2],
            self.goal_pose_local[:, 2],
            self.vehicle_pose_local[:, :2],
            self.vehicle_pose_local[:, 2],
        )
        distance = torch.linalg.norm(self.current_goal_obs[:, :2], dim=-1, keepdim=True)
        heading = torch.atan2(self.current_goal_obs[:, 2], self.current_goal_obs[:, 3]).unsqueeze(-1)
        self.kinematics_obs = torch.cat(
            [
                self.vehicle_state,
                self.last_action_seq[:, -1],
                distance,
                heading,
            ],
            dim=-1,
        )
        return {
            "camera_front": _camera(self.camera_front),
            "camera_left": _camera(self.camera_left),
            "camera_back": _camera(self.camera_back),
            "camera_right": _camera(self.camera_right),
            "goal": self.current_goal_obs.clone(),
            "kinematics": self.kinematics_obs.clone(),
        }

    def _get_rewards(self) -> torch.Tensor:
        self.current_goal_obs = self._transform_goal_to_body(
            self.goal_pose_local[:, :2],
            self.goal_pose_local[:, 2],
            self.vehicle_pose_local[:, :2],
            self.vehicle_pose_local[:, 2],
        )
        action_l2 = torch.mean(self.processed_action_seq.square(), dim=(1, 2))
        action_smoothness = torch.mean((self.processed_action_seq[:, 1:] - self.processed_action_seq[:, :-1]).square(), dim=(1, 2))
        near_goal = torch.linalg.norm(self.current_goal_obs[:, :2], dim=-1) < 1.5
        near_goal_stop = near_goal.float() * self.vehicle_state[:, 0].abs()

        reward = self._reward_step
        reward += self.cfg.reward_alive
        reward += self.cfg.reward_action_l2_weight * action_l2
        reward += self.cfg.reward_action_smoothness_weight * action_smoothness
        reward += self.cfg.reward_near_goal_stop_weight * near_goal_stop
        reward += self._collision_flag.float() * self.cfg.reward_collision
        reward += self._success_flag.float() * self.cfg.reward_success
        reward += self.reset_time_outs.float() * self.cfg.reward_timeout

        self.extras["log"] = {
            "reward_progress": self._reward_step.mean(),
            "collision_rate": self._collision_flag.float().mean(),
            "success_rate": self._success_flag.float().mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        scene_bounds = self.scene_local_bounds[self.scene_ids]
        x = self.vehicle_pose_local[:, 0]
        y = self.vehicle_pose_local[:, 1]
        out_of_bounds = (x < scene_bounds[:, 0, 0]) | (x > scene_bounds[:, 0, 1]) | (y < scene_bounds[:, 1, 0]) | (y > scene_bounds[:, 1, 1])
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._collision_flag | self._success_flag | out_of_bounds
        self.extras["time_out_penalty"] = time_out.float().mean() * self.cfg.reward_timeout
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        for env_id in env_ids.tolist():
            scene_id = int(self.scene_ids[env_id].item())
            goal_pose, spawn_pose = self._sample_spawn_pose(scene_id)
            self.goal_pose_local[env_id] = goal_pose
            self.vehicle_pose_local[env_id] = spawn_pose
            self.vehicle_state[env_id] = 0.0
            self.last_action_seq[env_id] = 0.0
            self.processed_action_seq[env_id] = 0.0
            self._reward_step[env_id] = 0.0
            self._collision_flag[env_id] = False
            self._success_flag[env_id] = False
            self._done_during_sequence[env_id] = False

        self._write_vehicle_pose_to_sim(env_ids)
