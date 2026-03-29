import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Jetbot-SpatialVerse-839920-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_spatialverse_env_cfg:JetbotSpatialVerse839920EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JetbotSpatialVerse839920PPORunnerCfg",
    },
)
