from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ..camera_calib import PARKING_CAMERA_KEYS, PARKING_CAMERA_MOUNTS, PARKING_LOOKUP_FISHEYE_CFG


@configclass
class ParkingFastRayModelCfg(RslRlMLPModelCfg):
    class_name = "isaaclab_tasks.direct.parking_end2end.agents.models:ParkingFastRayModel"
    camera_keys: list[str] = PARKING_CAMERA_KEYS
    goal_key: str = "goal"
    kinematics_key: str | None = None
    voxel_shape: tuple[int, int, int] = (8, 160, 120)
    voxel_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-1.0, 3.0), (12.0, -12.0), (9.0, -9.0))
    feature_downscale: int = 4
    camera_feature_channels: int = 80
    ray_distance_num_channel: int = 32
    ray_distance_start: float = 0.5
    ray_distance_step: float = 0.5
    ego_distance_max: float = 16.0
    ego_distance_step: float = 2.0
    goal_hidden_dims: list[int] = [64, 64]
    head_hidden_dims: list[int] = [256, 128]
    relu6: bool = True
    hidden_dims: list[int] = [256, 128]
    activation: str = "elu"
    camera_mounts: dict[str, dict] = PARKING_CAMERA_MOUNTS
    fisheye_cfg: dict[str, float] = PARKING_LOOKUP_FISHEYE_CFG


@configclass
class ParkingEnd2EndPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5000
    empirical_normalization = False
    save_interval = 50
    experiment_name = "parking_end2end_direct"
    obs_groups = {
        "actor": ["camera_front", "camera_left", "camera_back", "camera_right", "goal"],
        "critic": ["camera_front", "camera_left", "camera_back", "camera_right", "goal", "kinematics"],
    }
    clip_actions = 1.0
    actor = ParkingFastRayModelCfg(
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35, std_type="log"),
        kinematics_key=None,
    )
    critic = ParkingFastRayModelCfg(
        obs_normalization=False,
        distribution_cfg=None,
        kinematics_key="kinematics",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
