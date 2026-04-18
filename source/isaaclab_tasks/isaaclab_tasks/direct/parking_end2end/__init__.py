import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Parking-End2End-Direct-v0",
    entry_point=f"{__name__}.parking_env:ParkingEnd2EndEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parking_env:ParkingEnd2EndEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ParkingEnd2EndPPORunnerCfg",
    },
)
