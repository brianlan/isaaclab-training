# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test for the SpatialVerse Ant pilot scene.

Loads ``Isaac-Ant-SpatialVerse-839920-v0`` with a single environment,
applies **zero actions** for 200 steps, and prints root pose / root height
every 50 steps. Exits with a nonzero code on any of:

* Scene-load failure (exception during ``gym.make`` or ``env.reset``)
* NaN detected in observations or root state
* Root height dropping below 0.31 m

This script is the hard gate before PPO training (Task 9).

Usage
-----
.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/check_ant_spatialverse_scene.py --headless

Redirect stdout to capture evidence::

    ./isaaclab.sh -p scripts/tools/check_ant_spatialverse_scene.py --headless \
        > .sisyphus/evidence/task-8-scene-smoke.txt 2>&1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Smoke-test: load the SpatialVerse Ant scene and run 200 zero-action steps."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks  # noqa: F401 — triggers gym registrations
from isaaclab_tasks.utils import parse_env_cfg

TASK_ID = "Isaac-Ant-SpatialVerse-839920-v0"
NUM_STEPS = 200
LOG_INTERVAL = 50
MIN_ROOT_HEIGHT = 0.31


def _check_nan(tensor: torch.Tensor, label: str, step: int) -> bool:
    """Return True if any NaN is found. Prints error."""
    if torch.isnan(tensor).any():
        print(f"[FAIL] NaN detected in {label} at step {step}")
        return True
    return False


def main() -> int:
    """Run the smoke test. Returns 0 on success, 1 on failure."""

    print(f"[INFO] Creating environment: {TASK_ID} (num_envs=1)")
    try:
        env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
        env = gym.make(TASK_ID, cfg=env_cfg)
    except Exception as exc:
        print(f"[FAIL] Could not create environment {TASK_ID}: {exc}")
        return 1

    unwrapped: ManagerBasedRLEnv = env.unwrapped
    actual_num_envs = unwrapped.num_envs
    print(f"[INFO] num_envs = {actual_num_envs}")
    if actual_num_envs != 1:
        print(f"[WARN] Expected num_envs=1, got {actual_num_envs}. Continuing anyway.")

    print("[INFO] Resetting environment...")
    try:
        obs, info = env.reset()
    except Exception as exc:
        print(f"[FAIL] env.reset() raised: {exc}")
        env.close()
        return 1

    if _check_nan(obs["policy"], "initial observations", step=0):
        env.close()
        return 1

    action_shape = env.action_space.shape
    if action_shape is None:
        print("[FAIL] Action space has no shape — cannot determine action dimensions.")
        env.close()
        return 1
    print(f"[INFO] Action space shape: {action_shape}")

    print(f"[INFO] Running {NUM_STEPS} zero-action steps (logging every {LOG_INTERVAL})...")
    device = unwrapped.device

    zero_actions = torch.zeros(*action_shape, device=device)

    with torch.inference_mode():
        for step in range(1, NUM_STEPS + 1):
            if not simulation_app.is_running():
                print(f"[FAIL] Simulation app exited early at step {step}")
                env.close()
                return 1

            obs, reward, terminated, truncated, info = env.step(zero_actions)

            if _check_nan(obs["policy"], "observations", step):
                env.close()
                return 1

            root_pos = unwrapped.scene["robot"].data.root_pos_w
            root_quat = unwrapped.scene["robot"].data.root_quat_w
            root_height = root_pos[0, 2].item()

            if _check_nan(root_pos, "root_pos", step):
                env.close()
                return 1

            if root_height < MIN_ROOT_HEIGHT:
                print(
                    f"[FAIL] Root height {root_height:.4f} m < {MIN_ROOT_HEIGHT} m at step {step}. "
                    "Robot has fallen through or collapsed."
                )
                env.close()
                return 1

            if step % LOG_INTERVAL == 0 or step == 1:
                pos = root_pos[0].tolist()
                quat = root_quat[0].tolist()
                print(
                    f"[STEP {step:>4d}] "
                    f"root_pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  "
                    f"root_quat=({quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f})  "
                    f"root_height={root_height:.4f} m"
                )

    final_height = root_pos[0, 2].item()
    print("=" * 72)
    print(f"[PASS] Smoke test completed successfully after {NUM_STEPS} steps.")
    print(f"       Final root height: {final_height:.4f} m (>= {MIN_ROOT_HEIGHT} m)")
    print(f"       Task: {TASK_ID}")
    print("=" * 72)

    env.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
