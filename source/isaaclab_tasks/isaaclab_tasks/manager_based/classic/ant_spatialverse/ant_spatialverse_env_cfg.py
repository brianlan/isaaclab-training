# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import math

from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp


ant_env_cfg_module = importlib.import_module("isaaclab_tasks.manager_based.classic.ant.ant_env_cfg")
AntEnvCfg = ant_env_cfg_module.AntEnvCfg
MySceneCfg = ant_env_cfg_module.MySceneCfg
TerrainImporterCfg = importlib.import_module("isaaclab.terrains").TerrainImporterCfg
configclass = importlib.import_module("isaaclab.utils").configclass


SCENE_ID = "839920"
INTERIORGS_SCENE_DIR = "0001_839920"
INTERIORGS_ROOT = "/ssd5/datasets/InteriorGS"
SAGE_3D_ROOT = "/ssd5/datasets/SAGE-3D_Collision_Mesh"
SAGE_COLLISION_USD_PATH = f"{SAGE_3D_ROOT}/Collision_Mesh/{SCENE_ID}/{SCENE_ID}_collision.usd"

# Task 2 evidence-calibrated indoor pilot constants (hardcoded by design for Task 7 semantics).
CALIBRATED_SPAWN_CENTER_XYZ = (6.5, -2.0, 0.31)
CALIBRATED_TARGET_XYZ = (-1.0, -1.0, 0.31)
SPAWN_JITTER_X_M = 0.1
SPAWN_JITTER_Y_M = 1
SPAWN_YAW_DEG = 10.0
STOCK_ANT_INIT_Z_M = 0.5

robot_scale = 0.2


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31 * robot_scale})


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
        spawn=ANT_CFG.spawn.replace(scale=(robot_scale, robot_scale, robot_scale)),
    )


@configclass
class AntSpatialVerse839920EnvCfg(AntEnvCfg):
    scene: MySceneCfg = SpatialVerse839920SceneCfg(num_envs=1, env_spacing=0.1, clone_in_fabric=True)
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        spawn_x, spawn_y, spawn_z = CALIBRATED_SPAWN_CENTER_XYZ
        z_offset = spawn_z - STOCK_ANT_INIT_Z_M
        yaw_jitter_rad = math.radians(SPAWN_YAW_DEG)

        self.observations.policy.base_angle_to_target.params["target_pos"] = CALIBRATED_TARGET_XYZ
        self.observations.policy.base_heading_proj.params["target_pos"] = CALIBRATED_TARGET_XYZ
        self.rewards.progress.params["target_pos"] = CALIBRATED_TARGET_XYZ
        self.rewards.move_to_target.params["target_pos"] = CALIBRATED_TARGET_XYZ

        self.events.reset_base.params["pose_range"] = {
            "x": (spawn_x - SPAWN_JITTER_X_M, spawn_x + SPAWN_JITTER_X_M),
            "y": (spawn_y - SPAWN_JITTER_Y_M, spawn_y + SPAWN_JITTER_Y_M),
            "z": (z_offset, z_offset),
            "yaw": (-yaw_jitter_rad, yaw_jitter_rad),
        }

        self.episode_length_s = 20.0
