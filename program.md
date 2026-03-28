# ant_spatialverse_rl_autoresearch (robust mode v2)

Autonomous optimization loop for `Isaac-Ant-SpatialVerse-839920-v0` using a fixed evaluator, deterministic run contract, and safe keep/discard semantics.

## Setup

1. Create branch `autoresearch/<tag>` from `master`.
2. Use this Python environment for all commands:
   - `/ssd4/envs/isaac_sim_py311/bin/python`
3. Read these files:
   - `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/agents/rsl_rl_ppo_cfg.py` (editable)
   - `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_spatialverse/ant_spatialverse_env_cfg.py` (read-only during tuning loop)
   - `scripts/tools/check_ant_spatialverse_scene.py` (preflight)
   - `scripts/tools/evaluate_ant_spatialverse_checkpoint.py` (fixed evaluator)
   - `scripts/tools/extract_ant_spatialverse_run_metrics.py` (structured metrics)
4. Ensure preflight passes:
   ```bash
   CONDA_PREFIX=/ssd4/envs/isaac_sim_py311 PATH=/ssd4/envs/isaac_sim_py311/bin:$PATH \
   ./isaaclab.sh -p scripts/tools/check_ant_spatialverse_scene.py --headless
   ```
5. Create `results.tsv` with this header:
   ```tsv
   commit\tscore\tsuccess_rate\tcollision_rate\tgoal_reached_rate\ttrain_mean_reward\tmean_episode_length\tfps\tstatus\trun_dir\tdescription
   ```
6. Keep `results.tsv`, `run.log`, and generated run artifacts uncommitted.

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
score = success_rate - 0.25 * collision_rate
```

Evaluator command:

```bash
/ssd4/envs/isaac_sim_py311/bin/python scripts/tools/evaluate_ant_spatialverse_checkpoint.py \
  --checkpoint <checkpoint_path> \
  --topk 8
```

Important evaluator semantics:
- This evaluator reads TensorBoard scalars from the run directory that contains the checkpoint.
- It does **not** execute policy rollout from checkpoint weights.
- Therefore, the optimization target is a run-level proxy metric, not true checkpoint inference quality.

Secondary metrics (for diagnosis only, not selection criterion):
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Perf/total_fps`

Extract command:

```bash
/ssd4/envs/isaac_sim_py311/bin/python scripts/tools/extract_ant_spatialverse_run_metrics.py --log_dir <run_log_dir>
```

## Deterministic run contract

Every experiment must use:
- `--task Isaac-Ant-SpatialVerse-839920-v0`
- `--num_envs 64`
- `--max_iterations 100`
- `--seed 42`
- `--headless`

Operational invariants:
- Run from repository root: `/home/rlan/projects/vla-robot-demo/IsaacLab`
- Exactly one active ant training process at a time.
- Do not reuse partially-running previous jobs.

Timeout policy:
- Hard timeout: 20 minutes wall clock.
- On timeout, kill the full process group and mark `crash_timeout`.

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
     --max_iterations 100 \
     --seed 42 \
     --headless > run.log 2>&1
   ```
5. Parse run directory from logs:
   - `[INFO] Logging experiment in directory:`
   - `Exact experiment name requested from command line:`
6. Choose latest checkpoint in run (`model_*.pt`, highest iteration).
7. Run fixed evaluator and capture JSON.
8. Run structured metric extraction and capture JSON.
9. Append row to `results.tsv`.
10. If score improves, keep commit and update best pointers.
11. Else, rollback safely to best config state and mark `discard`.

## Crash handling

Mark as crash if any of the following:
- No checkpoint produced (`crash_no_checkpoint`)
- Evaluator output missing required fields (`crash_eval_invalid`)
- Training command non-zero exit (`crash_train_exit`)
- Hard timeout reached (`crash_timeout`)

Crash signature definition:
- Signature = normalized tuple of `(status, first_exception_line_or_error_token)`
- If 3 consecutive identical signatures occur: mark branch unhealthy and pause loop.

## Baseline and stop behavior

- First run is unmodified baseline under this exact contract.
- After baseline, continue loop until manually interrupted.
- Do not ask continuation prompts.

## Out-of-loop env stabilization policy

If repeated crashes or pathological terminations are traced to environment config (not PPO cfg), stop tuning loop and open a separate stabilization pass. That pass may modify `ant_spatialverse_env_cfg.py`, then restart autotuning from a fresh baseline.
