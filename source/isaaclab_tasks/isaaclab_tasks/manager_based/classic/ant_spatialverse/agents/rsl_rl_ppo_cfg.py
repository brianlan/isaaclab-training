# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntSpatialVerse839920PPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_spatialverse_839920"
    policy = AntPPORunnerCfg.policy.__class__(
        init_noise_std=AntPPORunnerCfg.policy.init_noise_std,
        actor_obs_normalization=AntPPORunnerCfg.policy.actor_obs_normalization,
        critic_obs_normalization=AntPPORunnerCfg.policy.critic_obs_normalization,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation=AntPPORunnerCfg.policy.activation,
    )
