SpatialVerse Ant Pilot
======================

This guide provides the runbook for the SpatialVerse Ant Pilot, a physics-only, single-scene simulation environment. The pilot focuses on integrating Ant robot dynamics with specific collision datasets in Isaac Lab.

The current implementation (as of 2026-03-26) validates end-to-end integration: dataset loading, scene construction, PPO training, and policy replay.

.. note::
   The 100-iteration pilot validated the software pipeline but did not yet show meaningful learning (episode rewards remained 0.0). This is expected given the low step count and restrictive termination conditions.

Prerequisites
-------------

*   **Isaac Sim Environment**: Ensure you are using the correct Python environment.
*   **Guardrail**: The Python executable at ``/ssd4/envs/isaac_sim_py311/bin/python`` is read-only. Do not attempt to modify this environment.
*   **Datasets**: Verify that the required datasets are available at the following paths:
    *   InteriorGS: ``/ssd5/datasets/InteriorGS/0001_839920``
    *   SAGE-3D Collision Mesh: ``/ssd5/datasets/SAGE-3D_Collision_Mesh/Collision_Mesh/839920``

Dataset Verification
--------------------

Before running training, verify the scene and dataset integration:

.. code-block:: bash

   # Run from the IsaacLab/ root directory
   ./isaaclab.sh -p scripts/tools/check_ant_spatialverse_scene.py --headless

This script confirms that the Ant robot can spawn and interact with the SAGE-3D collision mesh without immediate physics failure.

Scene Calibration
-----------------

The pilot uses a fixed spawn and target position derived from InteriorGS occupancy metadata. This ensures the robot starts on a walkable surface and has a clear path toward the goal.

Calibration was performed using the following script:

.. code-block:: bash

   # Path to the calibration script
   python IsaacLab/scripts/tools/calibrate_spatialverse_839920.py

**Methodology:**
1. **Inputs:** ``occupancy.png`` and ``occupancy.json``/``structure.json`` from the InteriorGS dataset.
2. **Analysis:** The script identifies the largest connected walkable component in the floor plan.
3. **Selection:** Spawn and target points are selected along the principal axis of this component (20th percentile for spawn, 80th percentile for target).
4. **Output:** The resulting coordinates are stored in ``.sisyphus/evidence/task-2-scene-calibration.json`` and hardcoded into the task configuration for this pilot.

Training
--------

Training uses the RSL RL reinforcement learning wrapper.

Smoke Test (5 Iterations)
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate the PPO loop with a minimal run:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=Isaac-Ant-SpatialVerse-839920-v0 \
     --headless --num_envs=1 --max_iterations=5 \
     --video --video_length=200

Pilot Run (100 Iterations)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute a longer run to verify checkpointing and video generation:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=Isaac-Ant-SpatialVerse-839920-v0 \
     --headless --num_envs=1 --max_iterations=100 \
     --video --video_length=300

Replay and Evaluation
---------------------

To replay the pilot policy (using the verified 100-iteration checkpoint):

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
     --task=Isaac-Ant-SpatialVerse-839920-v0 \
     --headless --num_envs=1 \
     --checkpoint IsaacLab/logs/rsl_rl/ant_spatialverse_839920/2026-03-26_17-12-16/model_99.pt \
     --video --video_length=600

Verified Artifacts
------------------

The pilot run on 2026-03-26 produced the following artifacts:

*   **Log Directory**: ``IsaacLab/logs/rsl_rl/ant_spatialverse_839920/2026-03-26_17-12-16/``
*   **Final Checkpoint**: ``IsaacLab/logs/rsl_rl/ant_spatialverse_839920/2026-03-26_17-12-16/model_99.pt``
*   **Replay Video**: ``IsaacLab/logs/rsl_rl/ant_spatialverse_839920/2026-03-26_17-12-16/videos/play/rl-video-step-0.mp4``

Troubleshooting
---------------

*   **Runtime success but rewards = 0.0**: Expected for short pilot runs (100 iterations). Policy convergence requires significantly more steps and environment instances.
*   **Unstable Spawn Height / Collision-Scene Instability**: 
    - **Symptom:** The robot starts too low (clipping the floor) or too high (falling and bouncing), triggering immediate ``torso_height`` termination.
    - **Cause:** Discrepancies between the InteriorGS metadata height and the SAGE-3D collision mesh.
    - **Fix:** Adjust the ``spawn_pos`` Z-coordinate in the task configuration to ensure the robot's base clears the floor mesh exactly without excessive drop.
*   **"clone_in_fabric" warning**: The error ``[Error] [isaacsim.core.cloner.impl.cloner] Failed to clone in Fabric`` is a known artifact during scene creation but does not block training or replay for this pilot.
*   **Missing Assets**: Verify paths to ``InteriorGS`` and ``SAGE-3D`` in the task configuration if URDF/USD loading fails.
*   **Environment Creation Failure (cfg=None)**: Ensure you use ``parse_env_cfg`` to generate the ``env_cfg`` object before calling ``gym.make``. Direct instantiation without config will fail.

Guardrails and Constraints
--------------------------

*   **Physics-Only**: This pilot is restricted to physics simulation; no visual rendering or camera sensors are included.
*   **Single-Scene**: The environment is limited to a single scene (839920) and a single environment instance for the pilot.
*   **Read-Only Environment**: ``/ssd4/envs/isaac_sim_py311/bin/python`` is protected. No package modifications should be performed.
