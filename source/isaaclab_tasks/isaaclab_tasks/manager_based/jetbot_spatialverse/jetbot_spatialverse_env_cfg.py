import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply


SCENE_ID = "839920"
SAGE_3D_ROOT_CANDIDATES = [
    Path("/ssd5/datasets/SAGE-3D_Collision_Mesh"),
    Path("/ssd5/datasets/SAGE3D/Collision_Mesh"),
]

CALIBRATED_SPAWN_CENTER_XYZ = (6.5, -2.0, 0.14)
CALIBRATED_TARGET_XYZ = (-1.8, 0.0, 0.14)
SPAWN_JITTER_X_M = 0.2
SPAWN_JITTER_Y_M = 1.0
SPAWN_YAW_DEG = 20.0
ROOT_FALL_MARGIN_M = 0.08
LIDAR_COLLISION_THRESHOLD_M = 0.14  # bot_radius(0.10m) + safety_margin(0.04m)

DEPTH_MAX_DIST = 10.0
LIDAR_CHANNELS = 16
LIDAR_VERT_FOV = (-35, 35)
LIDAR_HORIZ_FOV = (0, 360)
LIDAR_HORIZ_RES = 5


def resolve_collision_scene_path() -> str:
    candidate_paths = [root / f"Collision_Mesh/{SCENE_ID}/{SCENE_ID}_collision.usd" for root in SAGE_3D_ROOT_CANDIDATES]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(
        f"Could not locate collision USD for scene {SCENE_ID}. Checked: {[str(path) for path in candidate_paths]}"
    )


SAGE_COLLISION_USD_PATH = resolve_collision_scene_path()


JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
        activate_contact_sensors=True,
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.3, 1.0, 0.3)),  # bright green
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            damping=None,
            stiffness=None,
        )
    },
)


def root_height_below_spawn_margin(env, margin_m: float = ROOT_FALL_MARGIN_M) -> torch.Tensor:
    robot = env.scene["robot"]
    min_height = CALIBRATED_SPAWN_CENTER_XYZ[2] - margin_m
    return robot.data.root_pos_w[:, 2] < min_height


def _target_direction_xy(env, target_pos: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    target_xy = torch.tensor(target_pos[:2], device=env.device)
    root_xy = robot.data.root_pos_w[:, :2]
    target_vec = target_xy - root_xy
    target_dist = torch.norm(target_vec, dim=-1, keepdim=True)
    target_dir = target_vec / torch.clamp(target_dist, min=1.0e-6)
    return target_dir, target_dist


def distance_to_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    _, target_dist = _target_direction_xy(env, target_pos)
    return target_dist.squeeze(-1)


def heading_alignment_to_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    robot = env.scene["robot"]
    forward_world = quat_apply(robot.data.root_quat_w, robot.data.FORWARD_VEC_B)[:, :2]
    forward_world = forward_world / torch.clamp(torch.norm(forward_world, dim=-1, keepdim=True), min=1.0e-6)
    target_dir, _ = _target_direction_xy(env, target_pos)
    return torch.sum(forward_world * target_dir, dim=-1)


def turn_direction_to_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    robot = env.scene["robot"]
    forward_world = quat_apply(robot.data.root_quat_w, robot.data.FORWARD_VEC_B)
    forward_world = forward_world / torch.clamp(torch.norm(forward_world, dim=-1, keepdim=True), min=1.0e-6)
    target_dir_xy, _ = _target_direction_xy(env, target_pos)
    target_dir = torch.zeros_like(forward_world)
    target_dir[:, :2] = target_dir_xy
    return torch.cross(forward_world, target_dir, dim=-1)[:, 2]


def forward_speed_toward_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    robot = env.scene["robot"]
    target_dir, _ = _target_direction_xy(env, target_pos)
    lin_vel_xy = robot.data.root_lin_vel_w[:, :2]
    return torch.sum(lin_vel_xy * target_dir, dim=-1)


def obs_heading_alignment_to_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    return heading_alignment_to_target(env, target_pos=target_pos).unsqueeze(-1)


def obs_turn_direction_to_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    return turn_direction_to_target(env, target_pos=target_pos).unsqueeze(-1)


def obs_forward_speed_toward_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    return forward_speed_toward_target(env, target_pos=target_pos).unsqueeze(-1)


def obs_distance_to_target(env, target_pos: tuple[float, float, float]) -> torch.Tensor:
    return distance_to_target(env, target_pos=target_pos).unsqueeze(-1)


def yaw_rate_l2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.square(robot.data.root_ang_vel_w[:, 2])


def minimum_velocity_penalty(env, min_speed: float = 0.1) -> torch.Tensor:
    """Penalize when bot moves too slowly (below minimum speed threshold).

    This encourages the bot to keep moving rather than staying still.
    Returns 0 when speed >= min_speed, positive penalty when speed < min_speed.
    """
    robot = env.scene["robot"]
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    return torch.clamp(min_speed - speed, min=0.0)


def target_reached_terminated(env, target_pos: tuple[float, float, float], threshold: float = 0.45) -> torch.Tensor:
    return distance_to_target(env, target_pos) <= threshold


def goal_reached_reward(env, target_pos: tuple[float, float, float], threshold: float = 0.45) -> torch.Tensor:
    return target_reached_terminated(env, target_pos=target_pos, threshold=threshold).float()


def set_global_prim_visibility(env, env_ids, prim_path: str, visible: bool) -> None:
    prim = sim_utils.get_prim_at_path(prim_path)
    if prim is None or not prim.IsValid():
        raise ValueError(f"Could not resolve prim at path '{prim_path}' for visibility update.")
    sim_utils.set_prim_visibility(prim, visible)


def lidar_collision_terminated(env, sensor_cfg: SceneEntityCfg, threshold_m: float = LIDAR_COLLISION_THRESHOLD_M):
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=DEPTH_MAX_DIST, posinf=DEPTH_MAX_DIST, neginf=0.0)
    return torch.any(distances < threshold_m, dim=1)


def lidar_proximity_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    near_dist_m: float = 0.30,
    max_dist: float = DEPTH_MAX_DIST,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    return torch.mean(torch.clamp((near_dist_m - distances) / near_dist_m, min=0.0), dim=1)


def lidar_collision_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold_m: float = LIDAR_COLLISION_THRESHOLD_M,
    max_dist: float = DEPTH_MAX_DIST,
) -> torch.Tensor:
    """Returns -1.0 when collision occurs (min distance < threshold), else 0.0.

    This provides a sparse penalty when the bot is too close to obstacles,
    complementing the continuous proximity penalty.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    min_dist = distances.min(dim=1).values
    collision = (min_dist < threshold_m).float()  # 1.0 if collision, 0.0 otherwise
    return -collision  # -1.0 if collision, 0.0 otherwise


def depth_scan(env, sensor_cfg: SceneEntityCfg, max_dist: float = DEPTH_MAX_DIST):
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    return distances / max_dist


@configclass
class JetbotSpatialVerse839920SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=SAGE_COLLISION_USD_PATH,
        env_spacing=1.0,
        debug_vis=False,
    )

    robot = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    depth_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.10)),
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
                is_shared=True,
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            )
        ],
        max_distance=DEPTH_MAX_DIST,
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*wheel.*"], scale=20.0)


@configclass
class ObservationsCfg:
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
        actions = ObsTerm(func=mdp.last_action)
        depth_scan = ObsTerm(
            func=depth_scan,
            params={"sensor_cfg": SceneEntityCfg("depth_scanner"), "max_dist": DEPTH_MAX_DIST},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (
                    CALIBRATED_SPAWN_CENTER_XYZ[0] - SPAWN_JITTER_X_M,
                    CALIBRATED_SPAWN_CENTER_XYZ[0] + SPAWN_JITTER_X_M,
                ),
                "y": (
                    CALIBRATED_SPAWN_CENTER_XYZ[1] - SPAWN_JITTER_Y_M,
                    CALIBRATED_SPAWN_CENTER_XYZ[1] + SPAWN_JITTER_Y_M,
                ),
                "z": (0.0, 0.0),
                "yaw": (-math.radians(SPAWN_YAW_DEG), math.radians(SPAWN_YAW_DEG)),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    progress = RewTerm(func=forward_speed_toward_target, weight=2.5, params={"target_pos": CALIBRATED_TARGET_XYZ})
    # heading = RewTerm(func=heading_alignment_to_target, weight=1.0, params={"target_pos": CALIBRATED_TARGET_XYZ})  # removed: redundant with progress
    # distance = RewTerm(func=distance_to_target, weight=-0.2, params={"target_pos": CALIBRATED_TARGET_XYZ})  # removed: use progress (velocity-based distance delta) instead
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
        func=lidar_proximity_penalty,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("depth_scanner"),
            "near_dist_m": 0.40,
            "max_dist": DEPTH_MAX_DIST,
        },
    )
    # collision_terminated = RewTerm(
    #     func=lidar_collision_penalty,
    #     weight=-15.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("depth_scanner"),
    #         "threshold_m": LIDAR_COLLISION_THRESHOLD_M,
    #     },
    # )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.02})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
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
            "threshold": 0.45,
        },
    )


@configclass
class JetbotSpatialVerse839920EnvCfg(ManagerBasedRLEnvCfg):
    scene: JetbotSpatialVerse839920SceneCfg = JetbotSpatialVerse839920SceneCfg(
        num_envs=256, env_spacing=0.1, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 25.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2

        if self.scene.depth_scanner is not None:
            self.scene.depth_scanner.update_period = self.decimation * self.sim.dt
