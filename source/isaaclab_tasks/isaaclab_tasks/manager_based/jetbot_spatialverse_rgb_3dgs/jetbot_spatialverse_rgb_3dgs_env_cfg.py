import os
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..jetbot_spatialverse.jetbot_spatialverse_env_cfg import (
    ActionsCfg,
    CALIBRATED_TARGET_XYZ,
    EventCfg,
    JETBOT_CFG,
    forward_speed_toward_target,
    goal_reached_reward,
    minimum_velocity_penalty,
    obs_distance_to_target,
    obs_forward_speed_toward_target,
    obs_heading_alignment_to_target,
    obs_turn_direction_to_target,
    set_global_prim_visibility,
    target_reached_terminated,
    yaw_rate_l2,
)


SCENE_ID = "839920"
SAGE_3D_ROOT_CANDIDATES = [
    Path("/ssd5/datasets/SAGE-3D_Collision_Mesh"),
    Path("/ssd5/datasets/SAGE3D/Collision_Mesh"),
]
INTERIORGS_ROOT = Path(f"/ssd5/datasets/InteriorGS/0001_{SCENE_ID}")
MERGED_3DGS_COLLISION_USDA_PATH = Path(f"/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda/{SCENE_ID}.usda")
RGB_CAMERA_RESOLUTION = (32, 32)
RGB_FEATURE_RESOLUTION = (8, 8)


def resolve_collision_scene_path() -> str:
    candidate_paths = [root / f"Collision_Mesh/{SCENE_ID}/{SCENE_ID}_collision.usd" for root in SAGE_3D_ROOT_CANDIDATES]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(
        f"Could not locate collision USD for scene {SCENE_ID}. Checked: {[str(path) for path in candidate_paths]}"
    )


def resolve_visual_scene_path() -> str:
    override_path = os.environ.get("ISAACLAB_3DGS_VISUAL_USD_PATH")
    candidate_paths = []
    if override_path:
        candidate_paths.append(Path(override_path))

    candidate_paths.extend(
        [
            MERGED_3DGS_COLLISION_USDA_PATH,
            INTERIORGS_ROOT / "3dgs_converted.usdz",
            INTERIORGS_ROOT / "3dgs_converted.usd",
            INTERIORGS_ROOT / "3dgs_scene.usdz",
            INTERIORGS_ROOT / "3dgs_scene.usd",
            INTERIORGS_ROOT / "3dgs_compressed.usdz",
            INTERIORGS_ROOT / "3dgs_compressed.usd",
        ]
    )

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return str(candidate_path)

    warnings.warn(
        "No converted 3DGS USD/USDZ asset was found for scene 839920. "
        "Falling back to the collision USD for camera rendering. "
        "Set ISAACLAB_3DGS_VISUAL_USD_PATH to the converted visual asset to enable true 3DGS rendering.",
        stacklevel=2,
    )
    return SAGE_COLLISION_USD_PATH


def rgb_camera_embedding(
    env,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
    output_size: tuple[int, int] = RGB_FEATURE_RESOLUTION,
) -> torch.Tensor:
    images = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)[..., :3]
    image_tensor = images.permute(0, 3, 1, 2).float() / 255.0
    pooled = F.adaptive_avg_pool2d(image_tensor, output_size)
    return pooled.flatten(start_dim=1)


SAGE_COLLISION_USD_PATH = resolve_collision_scene_path()
VISUAL_SCENE_USD_PATH = resolve_visual_scene_path()
USE_SEPARATE_VISUAL_SCENE = VISUAL_SCENE_USD_PATH != SAGE_COLLISION_USD_PATH


@configclass
class JetbotSpatialVerse839920Rgb3dgsSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=SAGE_COLLISION_USD_PATH,
        env_spacing=1.0,
        debug_vis=False,
    )

    visual_scene = None

    if USE_SEPARATE_VISUAL_SCENE:
        visual_scene = AssetBaseCfg(
            prim_path="/World/visual_scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=VISUAL_SCENE_USD_PATH,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            collision_group=-1,
        )

    robot = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    rgb_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/rl_front_camera_sensor",
        update_period=0.0,
        height=RGB_CAMERA_RESOLUTION[0],
        width=RGB_CAMERA_RESOLUTION[1],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 25.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.12, 0.0, 0.10), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )

    collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        history_length=3,
        track_air_time=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class Rgb3dgsObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        heading_alignment = ObsTerm(func=obs_heading_alignment_to_target, params={"target_pos": CALIBRATED_TARGET_XYZ})
        turn_direction = ObsTerm(func=obs_turn_direction_to_target, params={"target_pos": CALIBRATED_TARGET_XYZ})
        forward_speed = ObsTerm(func=obs_forward_speed_toward_target, params={"target_pos": CALIBRATED_TARGET_XYZ})
        dist_to_target = ObsTerm(func=obs_distance_to_target, params={"target_pos": CALIBRATED_TARGET_XYZ})
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*wheel.*"])},
            scale=0.1,
        )
        rgb_embedding = ObsTerm(
            func=rgb_camera_embedding,
            params={
                "sensor_cfg": SceneEntityCfg("rgb_camera"),
                "data_type": "rgb",
                "output_size": RGB_FEATURE_RESOLUTION,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Rgb3dgsEventCfg(EventCfg):
    if USE_SEPARATE_VISUAL_SCENE:
        hide_collision_terrain = EventTerm(
            func=set_global_prim_visibility,
            mode="startup",
            params={"prim_path": "/World/ground", "visible": False},
        )


@configclass
class Rgb3dgsRewardsCfg:
    progress = RewTerm(func=forward_speed_toward_target, weight=2.5, params={"target_pos": CALIBRATED_TARGET_XYZ})
    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    goal_reached = RewTerm(
        func=goal_reached_reward,
        weight=5.0,
        params={"target_pos": CALIBRATED_TARGET_XYZ, "threshold": 0.45},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    yaw_rate = RewTerm(func=yaw_rate_l2, weight=-0.04)
    minimum_velocity = RewTerm(func=minimum_velocity_penalty, weight=-0.5, params={"min_speed": 0.1})
    collision_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 1.0},
    )


@configclass
class Rgb3dgsTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.02})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
    base_collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 5.0},
    )
    goal_reached = DoneTerm(
        func=target_reached_terminated,
        params={"target_pos": CALIBRATED_TARGET_XYZ, "threshold": 0.45},
    )


@configclass
class JetbotSpatialVerse839920Rgb3dgsEnvCfg(ManagerBasedRLEnvCfg):
    scene: JetbotSpatialVerse839920Rgb3dgsSceneCfg = JetbotSpatialVerse839920Rgb3dgsSceneCfg(
        num_envs=64, env_spacing=0.1, clone_in_fabric=False
    )
    observations: Rgb3dgsObservationsCfg = Rgb3dgsObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: Rgb3dgsRewardsCfg = Rgb3dgsRewardsCfg()
    terminations: Rgb3dgsTerminationsCfg = Rgb3dgsTerminationsCfg()
    events: Rgb3dgsEventCfg = Rgb3dgsEventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 25.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.num_rerenders_on_reset = 1
        self.sim.render.antialiasing_mode = "OFF"
        self.sim.render.enable_translucency = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_global_illumination = False
        self.sim.render.enable_shadows = False
        self.sim.render.enable_ambient_occlusion = False
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        if self.scene.rgb_camera is not None:
            self.scene.rgb_camera.update_period = self.decimation * self.sim.dt
