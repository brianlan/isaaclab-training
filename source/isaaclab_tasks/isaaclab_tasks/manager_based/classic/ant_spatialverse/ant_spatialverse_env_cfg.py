# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import math

import torch

from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp


ant_env_cfg_module = importlib.import_module("isaaclab_tasks.manager_based.classic.ant.ant_env_cfg")
AntEnvCfg = ant_env_cfg_module.AntEnvCfg
MySceneCfg = ant_env_cfg_module.MySceneCfg
TerrainImporterCfg = importlib.import_module("isaaclab.terrains").TerrainImporterCfg
configclass = importlib.import_module("isaaclab.utils").configclass


SCENE_ID = "839920"
SAGE_3D_ROOT = "/ssd5/datasets/SAGE-3D_Collision_Mesh"
SAGE_COLLISION_USD_PATH = f"{SAGE_3D_ROOT}/Collision_Mesh/{SCENE_ID}/{SCENE_ID}_collision.usd"

# Task 2 evidence-calibrated indoor pilot constants (hardcoded by design for Task 7 semantics).
CALIBRATED_SPAWN_CENTER_XYZ = (6.5, -2.0, 0.14)
CALIBRATED_TARGET_XYZ = (-1.0, -1.0, 0.14)
SPAWN_JITTER_X_M = 0.1
SPAWN_JITTER_Y_M = 1
SPAWN_YAW_DEG = 10.0
STOCK_ANT_INIT_Z_M = 0.5
ROOT_FALL_MARGIN_M = 0.12
LIDAR_COLLISION_THRESHOLD_M = 0.12

robot_scale = 0.2

# Depth scanner configuration constants
DEPTH_MAX_DIST = 10.0  # meters
LIDAR_CHANNELS = 16
LIDAR_VERT_FOV = (-45, 45)  # degrees
LIDAR_HORIZ_FOV = (0, 360)  # degrees
LIDAR_HORIZ_RES = 5  # degrees
# Total rays = ceil(360/5) * 16 = 72 * 16 = 1152


##
# Custom MDP functions
##


def root_height_below_spawn_margin(env, margin_m: float = ROOT_FALL_MARGIN_M) -> torch.Tensor:
    """Terminate when the root drops significantly below calibrated spawn height."""
    robot = env.scene["robot"]
    min_height = CALIBRATED_SPAWN_CENTER_XYZ[2] - margin_m
    return robot.data.root_pos_w[:, 2] < min_height


def lidar_collision_terminated(env, sensor_cfg: SceneEntityCfg, threshold_m: float = LIDAR_COLLISION_THRESHOLD_M):
    """Terminate when any LiDAR ray reports a near obstacle (collision proxy)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=DEPTH_MAX_DIST, posinf=DEPTH_MAX_DIST, neginf=0.0)
    return torch.any(distances < threshold_m, dim=1)


def lidar_proximity_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    near_dist_m: float = 0.35,
    max_dist: float = DEPTH_MAX_DIST,
) -> torch.Tensor:
    """Penalize proximity to obstacles using LiDAR distances."""
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    # Linear penalty in [0, 1] for rays closer than near_dist_m.
    return torch.mean(torch.clamp((near_dist_m - distances) / near_dist_m, min=0.0), dim=1)


def target_reached_terminated(env, target_pos: tuple[float, float, float], threshold: float = 0.6) -> torch.Tensor:
    robot = env.scene["robot"]
    target_xy = torch.tensor(target_pos[:2], device=env.device)
    root_xy = robot.data.root_pos_w[:, :2]
    return torch.norm(root_xy - target_xy, dim=-1) <= threshold


##
# Custom observation function
##


def depth_scan(env, sensor_cfg: SceneEntityCfg, max_dist: float = DEPTH_MAX_DIST):
    """Return 1D normalized distance array from a LiDAR-like ray caster sensor.

    The output shape is ``(num_envs, num_rays)`` with values in ``[0, 1]``,
    where 0 means the hit is at the sensor origin and 1 means no hit within
    *max_dist* (or the hit is at *max_dist*).
    """
    from isaaclab.sensors.ray_caster.ray_caster import RayCaster

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos = sensor.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)  # (num_envs, num_rays)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    return distances / max_dist


##
# Scene configuration
##


@configclass
class SpatialVerse839920SceneCfg(MySceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=SAGE_COLLISION_USD_PATH,
        env_spacing=1.0,
        debug_vis=False,
    )

    robot = ANT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ANT_CFG.spawn.replace(
            scale=(robot_scale, robot_scale, robot_scale),
            # activate_contact_sensors removed: Ant articulation links are not
            # compatible with PhysX create_rigid_body_view used by ContactSensor.
            # Collision detection is done via body_incoming_wrench instead.
        ),
    )

    # LiDAR-like depth scanner mounted on the robot torso for obstacle awareness.
    # Uses MultiMeshRayCaster because the USD terrain has multiple meshes under /Root.
    # The USD structure: /World/ground/terrain/Root/<multiple_mesh_objects>
    depth_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=LIDAR_CHANNELS,
            vertical_fov_range=LIDAR_VERT_FOV,
            horizontal_fov_range=LIDAR_HORIZ_FOV,
            horizontal_res=LIDAR_HORIZ_RES,
        ),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground/terrain",
                is_shared=True,  # Same mesh across all envs
                merge_prim_meshes=True,  # Merge all meshes
                track_mesh_transforms=False,  # Static mesh, no need to track
            )
        ],
        max_distance=DEPTH_MAX_DIST,
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=root_height_below_spawn_margin, params={"margin_m": ROOT_FALL_MARGIN_M})
    # (3) Terminate when LiDAR indicates immediate obstacle proximity (collision proxy).
    base_collision = DoneTerm(
        func=lidar_collision_terminated,
        params={
            "sensor_cfg": SceneEntityCfg("depth_scanner"),
            "threshold_m": LIDAR_COLLISION_THRESHOLD_M,
        },
    )
    goal_reached = DoneTerm(
        func=target_reached_terminated,
        params={
            "target_pos": CALIBRATED_TARGET_XYZ,
            "threshold": 0.6,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving toward target
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": CALIBRATED_TARGET_XYZ})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # (3) Reward for upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": CALIBRATED_TARGET_XYZ}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (6) Penalty for energy consumption
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
    )
    # (8) Penalty for obstacle proximity from LiDAR depth.
    collision_penalty = RewTerm(
        func=lidar_proximity_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("depth_scanner"),
            "near_dist_m": 0.35,
            "max_dist": DEPTH_MAX_DIST,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": CALIBRATED_TARGET_XYZ})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": CALIBRATED_TARGET_XYZ})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        # LiDAR depth scan for obstacle awareness (1152 rays, normalized to [0,1])
        depth_scan = ObsTerm(
            func=depth_scan,
            params={"sensor_cfg": SceneEntityCfg("depth_scanner"), "max_dist": DEPTH_MAX_DIST},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Environment configuration
##


@configclass
class AntSpatialVerse839920EnvCfg(AntEnvCfg):
    scene: MySceneCfg = SpatialVerse839920SceneCfg(num_envs=256, env_spacing=0.1, clone_in_fabric=False)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        spawn_x, spawn_y, spawn_z = CALIBRATED_SPAWN_CENTER_XYZ
        z_offset = spawn_z - STOCK_ANT_INIT_Z_M
        yaw_jitter_rad = math.radians(SPAWN_YAW_DEG)

        # Override observation target positions
        self.observations.policy.base_angle_to_target.params["target_pos"] = CALIBRATED_TARGET_XYZ
        self.observations.policy.base_heading_proj.params["target_pos"] = CALIBRATED_TARGET_XYZ

        # Override reward target positions
        self.rewards.progress.params["target_pos"] = CALIBRATED_TARGET_XYZ
        self.rewards.move_to_target.params["target_pos"] = CALIBRATED_TARGET_XYZ

        # Reset event configuration
        self.events.reset_base.params["pose_range"] = {
            "x": (spawn_x - SPAWN_JITTER_X_M, spawn_x + SPAWN_JITTER_X_M),
            "y": (spawn_y - SPAWN_JITTER_Y_M, spawn_y + SPAWN_JITTER_Y_M),
            "z": (z_offset, z_offset),
            "yaw": (-yaw_jitter_rad, yaw_jitter_rad),
        }

        self.episode_length_s = 20.0

        # Sensor update period for depth scanner
        if self.scene.depth_scanner is not None:
            self.scene.depth_scanner.update_period = self.decimation * self.sim.dt
