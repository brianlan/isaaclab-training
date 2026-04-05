import argparse
import glob
import json
import os

from tensorboard.backend.event_processing import event_accumulator


parser = argparse.ArgumentParser(description="Extract structured metrics from an IsaacLab RSL-RL run directory.")
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--topk", type=int, default=8)
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()


def latest_event_file(log_dir: str) -> str:
    candidates = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard events file found in {log_dir}")
    return max(candidates, key=os.path.getctime)


def scalar_series(ea: event_accumulator.EventAccumulator, tag: str) -> list[float]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [x.value for x in ea.Scalars(tag)]


def latest_value(values: list[float]):
    return values[-1] if values else None


def topk_mean(values: list[float], k: int):
    if not values:
        return None
    k = min(max(1, k), len(values))
    return sum(sorted(values)[-k:]) / k


event_file = latest_event_file(args.log_dir)
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

mean_reward = scalar_series(ea, "Train/mean_reward")
mean_ep_len = scalar_series(ea, "Train/mean_episode_length")
total_fps = scalar_series(ea, "Perf/total_fps")

collision_term = scalar_series(ea, "Episode_Termination/base_collision")
goal_term = scalar_series(ea, "Episode_Termination/goal_reached")

result = {
    "log_dir": os.path.abspath(args.log_dir),
    "event_file": os.path.basename(event_file),
    "train_mean_reward_last": latest_value(mean_reward),
    "train_mean_reward_topk_mean": topk_mean(mean_reward, args.topk),
    "train_mean_episode_length_last": latest_value(mean_ep_len),
    "perf_total_fps_last": latest_value(total_fps),
    "episode_termination_base_collision_last": latest_value(collision_term),
    "episode_termination_goal_reached_last": latest_value(goal_term),
    "num_reward_points": len(mean_reward),
}

out_text = json.dumps(result, indent=2)
print(out_text)

if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text + "\n")
