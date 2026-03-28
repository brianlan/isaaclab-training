import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Ant-SpatialVerse-839920-v0")
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    obs, info = env.reset()

    robot = unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0]

    # feet configured in this task
    foot_names = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
    foot_ids = [robot.body_names.index(n) for n in foot_names]
    foot_pos = robot.data.body_pos_w[0, foot_ids]

    print(f"PROBE root_pos_w xyz: {root_pos.tolist()}")
    print(f"PROBE root_z: {root_pos[2].item():.6f}")
    print(f"PROBE foot_zs: {[round(v, 6) for v in foot_pos[:, 2].tolist()]}")
    print(f"PROBE min_foot_z: {foot_pos[:, 2].min().item():.6f}")
    print(f"PROBE root_minus_min_foot: {(root_pos[2]-foot_pos[:, 2].min()).item():.6f}")

    # lidar shortest hit distance as collision-proxy sanity
    sensor = unwrapped.scene.sensors["depth_scanner"]
    hit_positions = sensor.data.ray_hits_w[0]
    sensor_pos = sensor.data.pos_w[0]
    distances = torch.norm(hit_positions - sensor_pos.unsqueeze(0), dim=-1)
    distances = torch.nan_to_num(distances, nan=10.0, posinf=10.0, neginf=0.0)
    print(f"PROBE lidar_min_dist: {distances.min().item():.6f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
