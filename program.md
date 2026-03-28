# ant_spatialverse_rl_autoresearch (rollout metric mode v3)

Autonomous optimization loop for `Isaac-Ant-SpatialVerse-839920-v0` using deterministic training and rollout-based checkpoint evaluation.

## Setup

1. Create branch `autoresearch/<tag>` from `master`.
2. Use this Python environment for all commands:
   - `/ssd4/envs/isaac_sim_py311/bin/python`
3. Read these files:
   - `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/agents/rsl_rl_ppo_cfg.py` (editable)
   - `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/ant_spatialverse_env_cfg.py` (read-only during tuning loop)
   - `scripts/tools/check_ant_spatialverse_scene.py` (preflight)
   - `scripts/tools/evaluate_ant_spatialverse_checkpoint.py` (fixed rollout evaluator)
   - `scripts/tools/extract_ant_spatialverse_run_metrics.py` (training diagnostics only)
4. Ensure preflight passes:
   ```bash
   CONDA_PREFIX=/ssd4/envs/isaac_sim_py311 PATH=/ssd4/envs/isaac_sim_py311/bin:$PATH \
   ./isaaclab.sh -p scripts/tools/check_ant_spatialverse_scene.py --headless
   ```
5. Create `results.tsv` with this header:
   ```tsv
   commit	score	success_rate	collision_rate	timeout_rate	mean_episode_length	mean_episode_return	train_mean_reward	train_mean_episode_length	fps	status	run_dir	checkpoint	description
   ```
6. Keep `results.tsv`, `run.log`, `eval.log`, and generated run artifacts uncommitted.

## Scope and boundaries

### Can modify
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/agents/rsl_rl_ppo_cfg.py`

### Cannot modify (within tuning loop)
- `scripts/tools/evaluate_ant_spatialverse_checkpoint.py`
- `scripts/tools/extract_ant_spatialverse_run_metrics.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`
- Scene assets / USD / core IsaacLab framework

## Fixed objective and evaluator semantics

Primary score (higher is better):

```text
score = success_rate - 0.5 * collision_rate - 0.1 * timeout_rate
```

Inner-loop evaluator command (used every trial):

```bash
CONDA_PREFIX=/ssd4/envs/isaac_sim_py311 PATH=/ssd4/envs/isaac_sim_py311/bin:$PATH \
./isaaclab.sh -p scripts/tools/evaluate_ant_spatialverse_checkpoint.py \
  --task Isaac-Ant-SpatialVerse-839920-v0 \
  --checkpoint <checkpoint_path> \
  --headless \
  --num_envs 8 \
  --num_episodes 32 \
  --eval_seeds 42,43
```

Full-evaluation command (run only for promising candidates, e.g. inner-loop score improvement >= 0.01):

```bash
CONDA_PREFIX=/ssd4/envs/isaac_sim_py311 PATH=/ssd4/envs/isaac_sim_py311/bin:$PATH \
./isaaclab.sh -p scripts/tools/evaluate_ant_spatialverse_checkpoint.py \
  --task Isaac-Ant-SpatialVerse-839920-v0 \
  --checkpoint <checkpoint_path> \
  --headless \
  --num_envs 8 \
  --num_episodes 128 \
  --eval_seeds 42,43,44
```

Evaluator outputs required fields:
- `score`
- `success_rate`
- `collision_rate`
- `timeout_rate`
- `mean_episode_length`
- `mean_episode_return`
- `collected_episodes`

Secondary diagnostics (not primary selection criterion):
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Perf/total_fps`

Diagnostics extraction command:

```bash
/ssd4/envs/isaac_sim_py311/bin/python scripts/tools/extract_ant_spatialverse_run_metrics.py --log_dir <run_log_dir>
```

## Deterministic run contract

Every experiment must use:
- `--task Isaac-Ant-SpatialVerse-839920-v0`
- `--num_envs 64`
- `--max_iterations 1000`
- `--seed 42`
- `--headless`

Operational invariants:
- Run from repository root: `/home/rlan/projects/vla-robot-demo/IsaacLab`
- Exactly one active ant training process at a time.
- Do not reuse partially-running previous jobs.

Timeout policy:
- Hard timeout: 20 minutes wall clock.
- On timeout, kill the full process group and mark `crash_timeout`.

## Keep/discard rule

1. Accept if `score` strictly improves by at least `0.005`.
2. If `|score_delta| < 0.005`, apply tie-breakers in order:
   - higher `success_rate`
   - lower `collision_rate`
   - higher `mean_episode_return`
   - higher `mean_episode_length` (only if collision does not increase)
3. Reject if `collision_rate >= 0.98` unless `success_rate` improves by at least `0.05` over best.
4. Reject if evaluator collected fewer than 90% of requested episodes.

## Safe keep/discard semantics

Before each iteration:
1. Require clean git tree (except allowed untracked logs/results files).
2. Record `best_commit` and `best_score` from `results.tsv`.

On non-improving run:
- Prefer `git checkout <best_commit> -- <editable_file>` for surgical rollback.
- Use `git reset --hard <best_commit>` only if tree is clean and no concurrent work exists.
- Never destroy unrelated user changes.

## Experiment loop

1. Check repo cleanliness and single-process invariant.
2. Edit only `rsl_rl_ppo_cfg.py` with one clear hypothesis.
3. Commit the change with concise hypothesis in message.
4. Train:
   ```bash
   CONDA_PREFIX=/ssd4/envs/isaac_sim_py311 PATH=/ssd4/envs/isaac_sim_py311/bin:$PATH \
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Ant-SpatialVerse-839920-v0 \
     --num_envs 64 \
     --max_iterations 1000 \
     --seed 42 \
     --headless > run.log 2>&1
   ```
5. Parse run directory from logs:
   - `[INFO] Logging experiment in directory:`
   - `Exact experiment name requested from command line:`
6. Choose latest checkpoint in run (`model_*.pt`, highest iteration).
7. Run rollout evaluator and capture JSON in `eval.log`.
8. Run structured metric extraction and capture JSON.
9. Append row to `results.tsv`.
10. If accepted by rule, keep commit and update best pointers.
11. Else rollback safely to best config state and mark `discard`.

## Crash handling

Mark as crash if any of the following:
- No checkpoint produced (`crash_no_checkpoint`)
- Evaluator output missing required fields (`crash_eval_invalid`)
- Training command non-zero exit (`crash_train_exit`)
- Hard timeout reached (`crash_timeout`)
- Rollout evaluator runtime error (`crash_eval_runtime`)

Crash signature definition:
- Signature = normalized tuple of `(status, first_exception_line_or_error_token)`
- If 3 consecutive identical signatures occur: mark branch unhealthy and pause loop.

## Baseline and stop behavior

- First run is unmodified baseline under this exact contract.
- After baseline, continue loop until manually interrupted.
- Do not ask continuation prompts.

## Out-of-loop env stabilization policy

If repeated crashes or pathological terminations are traced to environment config (not PPO cfg), stop tuning loop and open a separate stabilization pass. That pass may modify `ant_spatialverse_env_cfg.py`, then restart autotuning from a fresh baseline.
