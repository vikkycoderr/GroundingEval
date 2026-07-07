# GroundingEval
Claim-level grounding evaluation for TeLLMe scene understanding.

## Local data setup

Raw CARLA and Smart Room inputs should live locally under `data/`. The folder is ignored by git except for `.gitkeep` placeholders, so large videos, frames, lidar, and sensor dumps stay off GitHub.

Copy or symlink a CARLA simulation into:

```text
data/carla/simulation_45/
```

Then run InternVL3-9B with video-first input:

```bash
python3 scripts/run_vlm_on_simulation.py --sim-dir data/carla/simulation_45 --scene-id carla_simulation_45 --input-mode auto --model-path /home/native/internvl3/InternVL/pretrained/InternVL3-9B
```

Generated GroundingEval outputs are written under `scenarios/`.
