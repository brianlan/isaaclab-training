import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rollout-based evaluator for Isaac-Ant-SpatialVerse checkpoints.")
parser.add_argument("--task", type=str, default="Isaac-Ant-SpatialVerse-839920-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8, help="Parallel envs for evaluation rollout.")
parser.add_argument("--num_episodes", type=int, default=32, help="Total episodes to collect across all seeds.")
parser.add_argument(
    "--eval_seeds",
    type=str,
    default="42,43",
    help="Comma-separated integer seeds. Episodes are split as evenly as possible across these seeds.",
)
parser.add_argument(
    "--goal_term_tag",
    type=str,
    default="Episode_Termination/goal_reached",
    help="Termination metric key for task success in extras['log'].",
)
parser.add_argument(
    "--collision_term_tag",
    type=str,
    default="Episode_Termination/base_collision",
    help="Termination metric key for collision failure in extras['log'].",
)
parser.add_argument(
    "--timeout_term_tag",
    type=str,
    default="Episode_Termination/time_out",
    help="Termination metric key for timeout in extras['log'] (if available).",
)
parser.add_argument("--collision_penalty_weight", type=float, default=0.5)
parser.add_argument("--timeout_penalty_weight", type=float, default=0.1)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")


@dataclass
class EpisodeStats:
    episodes: int = 0
    goal_count: float = 0.0
    collision_count: float = 0.0
    timeout_count: float = 0.0
    mean_episode_length: float = 0.0
    mean_episode_return: float = 0.0


@dataclass
class EvalSummary:
    checkpoint: str
    task: str
    seeds: list[int]
    requested_episodes: int
    collected_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_episode_length: float
    mean_episode_return: float
    score: float
    collision_penalty_weight: float
    timeout_penalty_weight: float
    per_seed: list[dict]


def _parse_seeds(seed_text: str) -> list[int]:
    seeds = []
    for tok in seed_text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    if not seeds:
        raise ValueError("--eval_seeds must contain at least one integer seed")
    return seeds


def _alloc_episodes(total: int, n_bins: int) -> list[int]:
    base = total // n_bins
    rem = total % n_bins
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def _rollout_seed(
    *,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    seed: int,
    episodes_to_collect: int,
    checkpoint_path: str,
) -> dict:
    env_cfg.seed = seed
    agent_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic
    else:
        policy_nn = None

    obs = env.get_observations()

    ep_returns = torch.zeros(env.num_envs, device=env.device)
    ep_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    goal_count = 0.0
    collision_count = 0.0
    timeout_count = 0.0

    with torch.inference_mode():
        while len(completed_returns) < episodes_to_collect and simulation_app.is_running():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            ep_returns += rewards
            ep_lengths += 1

            done_mask = dones.to(dtype=torch.bool)
            num_done = int(done_mask.sum().item())

            if num_done > 0:
                done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
                remaining = episodes_to_collect - len(completed_returns)
                take_n = min(remaining, num_done)
                take_indices = done_indices[:take_n]

                completed_returns.extend(ep_returns[take_indices].detach().cpu().tolist())
                completed_lengths.extend(ep_lengths[take_indices].detach().cpu().tolist())

                log_dict = extras.get("log", {}) if isinstance(extras, dict) else {}
                goal_avg = _to_float(log_dict.get(args_cli.goal_term_tag, 0.0))
                collision_avg = _to_float(log_dict.get(args_cli.collision_term_tag, 0.0))
                timeout_avg = _to_float(log_dict.get(args_cli.timeout_term_tag, 0.0))
                goal_count += goal_avg * take_n
                collision_count += collision_avg * take_n
                timeout_count += timeout_avg * take_n

                ep_returns[done_indices] = 0.0
                ep_lengths[done_indices] = 0

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

    env.close()

    stats = EpisodeStats(
        episodes=len(completed_returns),
        goal_count=goal_count,
        collision_count=collision_count,
        timeout_count=timeout_count,
        mean_episode_length=_mean(completed_lengths),
        mean_episode_return=_mean(completed_returns),
    )
    return asdict(stats)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    checkpoint_path = os.path.abspath(retrieve_file_path(args_cli.checkpoint))
    seeds = _parse_seeds(args_cli.eval_seeds)
    episodes_per_seed = _alloc_episodes(args_cli.num_episodes, len(seeds))

    per_seed = []
    total_episodes = 0
    total_goal = 0.0
    total_collision = 0.0
    total_timeout = 0.0
    weighted_len = 0.0
    weighted_ret = 0.0

    for seed, n_eps in zip(seeds, episodes_per_seed, strict=True):
        if n_eps <= 0:
            continue
        seed_stats = _rollout_seed(
            env_cfg=env_cfg.copy(),
            agent_cfg=agent_cfg.copy(),
            seed=seed,
            episodes_to_collect=n_eps,
            checkpoint_path=checkpoint_path,
        )
        seed_stats["seed"] = seed
        per_seed.append(seed_stats)

        total_episodes += seed_stats["episodes"]
        total_goal += seed_stats["goal_count"]
        total_collision += seed_stats["collision_count"]
        total_timeout += seed_stats["timeout_count"]
        weighted_len += seed_stats["mean_episode_length"] * seed_stats["episodes"]
        weighted_ret += seed_stats["mean_episode_return"] * seed_stats["episodes"]

    if total_episodes <= 0:
        raise RuntimeError("No episodes were collected during rollout evaluation.")

    success_rate = total_goal / total_episodes
    collision_rate = total_collision / total_episodes
    timeout_rate = total_timeout / total_episodes
    mean_episode_length = weighted_len / total_episodes
    mean_episode_return = weighted_ret / total_episodes

    score = success_rate - args_cli.collision_penalty_weight * collision_rate - args_cli.timeout_penalty_weight * timeout_rate

    summary = EvalSummary(
        checkpoint=checkpoint_path,
        task=args_cli.task,
        seeds=seeds,
        requested_episodes=args_cli.num_episodes,
        collected_episodes=total_episodes,
        success_rate=success_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        mean_episode_length=mean_episode_length,
        mean_episode_return=mean_episode_return,
        score=score,
        collision_penalty_weight=args_cli.collision_penalty_weight,
        timeout_penalty_weight=args_cli.timeout_penalty_weight,
        per_seed=per_seed,
    )

    out_text = json.dumps(asdict(summary), indent=2)
    print(out_text)
    if args_cli.output:
        with open(args_cli.output, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
