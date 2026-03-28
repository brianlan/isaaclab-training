import argparse
import importlib.metadata as metadata
import json
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate ant_spatialverse checkpoint with fixed metrics.")
parser.add_argument("--task", type=str, default="Isaac-Ant-SpatialVerse-839920-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_eval_episodes", type=int, default=128)
parser.add_argument("--max_total_steps", type=int, default=200000)
parser.add_argument("--goal_x", type=float, default=-1.0)
parser.add_argument("--goal_y", type=float, default=-1.0)
parser.add_argument("--goal_radius", type=float, default=0.6)
parser.add_argument("--collision_force_threshold", type=float, default=5.0)
parser.add_argument("--collision_penalty_weight", type=float, default=0.25)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from packaging import version

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


installed_version = metadata.version("rsl-rl-lib")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    target_xy = torch.tensor([args_cli.goal_x, args_cli.goal_y], device=device)
    torso_idx = env.unwrapped.scene["robot"].find_bodies("torso")[0][0]

    obs = env.get_observations()

    ep_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    ep_goal_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    ep_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=device)

    episodes = []
    total_steps = 0

    while (
        simulation_app.is_running()
        and len(episodes) < args_cli.num_eval_episodes
        and total_steps < args_cli.max_total_steps
    ):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            total_steps += 1

            robot = env.unwrapped.scene["robot"]
            root_xy = robot.data.root_pos_w[:, :2]
            dist_xy = torch.norm(root_xy - target_xy, dim=-1)
            ep_goal_reached |= dist_xy <= args_cli.goal_radius

            wrench = robot.data.body_incoming_joint_wrench_b[:, torso_idx, :3]
            force_mag = torch.norm(wrench, dim=-1)
            ep_collision |= force_mag > args_cli.collision_force_threshold

            ep_steps += 1

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                time_outs = extras.get("time_outs")
                for env_id in done_ids.tolist():
                    if len(episodes) >= args_cli.num_eval_episodes:
                        break
                    timed_out = bool(time_outs[env_id].item()) if time_outs is not None else False
                    collision = bool(ep_collision[env_id].item())
                    reached = bool(ep_goal_reached[env_id].item())
                    success = bool(reached and not collision)
                    episodes.append(
                        {
                            "success": success,
                            "collision": collision,
                            "goal_reached": reached,
                            "timed_out": timed_out,
                            "steps": int(ep_steps[env_id].item()),
                        }
                    )
                    ep_collision[env_id] = False
                    ep_goal_reached[env_id] = False
                    ep_steps[env_id] = 0

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

    env.close()

    if len(episodes) == 0:
        result = {
            "num_episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "goal_reached_rate": 0.0,
            "mean_episode_steps": 0.0,
            "score": -1.0,
            "error": "no_episodes_completed",
        }
    else:
        success_rate = sum(1 for x in episodes if x["success"]) / len(episodes)
        collision_rate = sum(1 for x in episodes if x["collision"]) / len(episodes)
        goal_reached_rate = sum(1 for x in episodes if x["goal_reached"]) / len(episodes)
        mean_episode_steps = sum(x["steps"] for x in episodes) / len(episodes)
        score = success_rate - args_cli.collision_penalty_weight * collision_rate

        result = {
            "num_episodes": len(episodes),
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "goal_reached_rate": goal_reached_rate,
            "mean_episode_steps": mean_episode_steps,
            "score": score,
            "goal_xy": [args_cli.goal_x, args_cli.goal_y],
            "goal_radius": args_cli.goal_radius,
            "collision_force_threshold": args_cli.collision_force_threshold,
            "collision_penalty_weight": args_cli.collision_penalty_weight,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "checkpoint": os.path.abspath(args_cli.checkpoint),
        }

    out_text = json.dumps(result, indent=2)
    print(out_text)
    if args_cli.output:
        with open(args_cli.output, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
