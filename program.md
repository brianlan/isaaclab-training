# ant_spatialverse_rl_autoresearch (robust mode)

Autonomous optimization loop for `Isaac-Ant-SpatialVerse-839920-v0` using a fixed evaluator, deterministic run contract, and keep/discard reset semantics.

## Setup

1. Pick a run tag (`mar28`) and create branch `autoresearch/<tag>` from master.
2. Read these files:
   - `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/agents/rsl_rl_ppo_cfg.py` (editable)
   - `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/ant_spatialverse_env_cfg.py` (read-only for loop stability)
   - `scripts/tools/check_ant_spatialverse_scene.py` (preflight)
   - `scripts/tools/evaluate_ant_spatialverse_checkpoint.py` (fixed evaluator)
   - `scripts/tools/extract_ant_spatialverse_run_metrics.py` (structured run metrics)
3. Ensure preflight passes:
   ```bash
   ./isaaclab.sh -p scripts/tools/check_ant_spatialverse_scene.py --headless
   ```
4. Create `results.tsv` with this header:
   ```tsv
   commit	score	success_rate	collision_rate	goal_reached_rate	train_mean_reward	fps	status	description
   ```

## Scope and boundaries

### Can modify
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/agents/rsl_rl_ppo_cfg.py`

### Cannot modify
- `scripts/tools/evaluate_ant_spatialverse_checkpoint.py`
- `scripts/tools/extract_ant_spatialverse_run_metrics.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`
- Scene assets / USD / core IsaacLab framework

## Fixed objective (read-only evaluator)

Primary score (higher is better):

```text
score = success_rate - 0.25 * collision_rate
```

Evaluator command:

```bash
./isaaclab.sh -p scripts/tools/evaluate_ant_spatialverse_checkpoint.py \
  --task Isaac-Ant-SpatialVerse-839920-v0 \
  --checkpoint <checkpoint_path> \
  --num_envs 32 \
  --num_eval_episodes 128 \
  --seed 42 \
  --headless
```

Secondary metrics (not selection criterion):
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Perf/total_fps`

Extract command:

```bash
python scripts/tools/extract_ant_spatialverse_run_metrics.py --log_dir <run_log_dir>
```

## Deterministic run contract

Every experiment must use:
- `--task Isaac-Ant-SpatialVerse-839920-v0`
- `--num_envs 64`
- `--max_iterations 100`
- `--seed 42`
- `--headless`

Timeout policy:
- hard kill if wall clock > 20 min
- mark as `crash`

## Experiment loop

1. Check git state, record current best commit and best score.
2. Edit only `rsl_rl_ppo_cfg.py` with one clear hypothesis.
3. Commit the change.
4. Train:
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Ant-SpatialVerse-839920-v0 \
     --num_envs 64 \
     --max_iterations 100 \
     --seed 42 \
     --headless > run.log 2>&1
   ```
5. Parse training run directory from logs (`[INFO] Logging experiment in directory:` + `Exact experiment name requested from command line:`).
6. Pick latest checkpoint in that run (`model_*.pt`, highest iteration).
7. Run fixed evaluator on checkpoint, capture JSON.
8. Run structured metric extraction on run directory.
9. Append TSV row.
10. If `score` improved over best-so-far, keep commit and advance best pointer.
11. Else: `git reset --hard <best_commit>` and mark `discard`.

## Crash handling

- If no checkpoint produced: mark `crash`.
- If evaluator returns zero completed episodes: mark `crash`.
- If 3 consecutive crashes share same signature: pause loop and mark branch unhealthy.

## Baseline and never-stop behavior

- First run is unmodified baseline.
- After baseline, loop forever until manually interrupted.
- Do not ask for continuation prompts.
