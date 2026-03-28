import argparse
import glob
import json
import os

from tensorboard.backend.event_processing import event_accumulator


parser = argparse.ArgumentParser(description="Evaluate ant_spatialverse checkpoint from fixed TensorBoard metrics.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--topk", type=int, default=8)
parser.add_argument("--collision_penalty_weight", type=float, default=0.25)
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()


def latest_event_file(run_dir: str) -> str:
    candidates = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard events file found in {run_dir}")
    return max(candidates, key=os.path.getctime)


def scalar_series(ea: event_accumulator.EventAccumulator, tag: str) -> list[float]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [x.value for x in ea.Scalars(tag)]


def last(values: list[float]):
    return values[-1] if values else None


def topk_mean(values: list[float], k: int):
    if not values:
        return None
    k = min(max(1, k), len(values))
    return sum(sorted(values)[-k:]) / k


checkpoint_abs = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(checkpoint_abs)
event_file = latest_event_file(run_dir)

ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

goal_series = scalar_series(ea, "Episode_Termination/goal_reached")
collision_series = scalar_series(ea, "Episode_Termination/base_collision")
reward_series = scalar_series(ea, "Train/mean_reward")
len_series = scalar_series(ea, "Train/mean_episode_length")
fps_series = scalar_series(ea, "Perf/total_fps")

success_rate = last(goal_series)
collision_rate = last(collision_series)

if success_rate is None:
    success_rate = 0.0
if collision_rate is None:
    collision_rate = 1.0

score = success_rate - args.collision_penalty_weight * collision_rate

result = {
    "checkpoint": checkpoint_abs,
    "run_dir": run_dir,
    "event_file": os.path.basename(event_file),
    "success_rate": success_rate,
    "collision_rate": collision_rate,
    "goal_reached_rate": success_rate,
    "score": score,
    "collision_penalty_weight": args.collision_penalty_weight,
    "train_mean_reward_last": last(reward_series),
    "train_mean_reward_topk_mean": topk_mean(reward_series, args.topk),
    "train_mean_episode_length_last": last(len_series),
    "perf_total_fps_last": last(fps_series),
}

out_text = json.dumps(result, indent=2)
print(out_text)
if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text + "\n")
