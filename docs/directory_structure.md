# Directory Structure

GroundingEval is organized as a multimodal benchmark for CARLA simulations and
Smart Room camera/sensor recordings. Dataset-specific adapters prepare raw data
for a shared scenario, QA, claim extraction, validation, scoring, and reporting
pipeline.

## Repository Layout

```text
GroundingEval/
├── adapters/
│   ├── carla_adapter.py
│   └── smartroom_adapter.py
├── data/
│   ├── carla/
│   └── smartroom/
├── scripts/
│   ├── score_claims.py
│   ├── validate_ground_truth.py
│   ├── run_vlm_on_simulation.py
│   ├── generate_qa_pairs.py
│   └── extract_claims.py
├── scenarios/
│   ├── carla/
│   └── smartroom_sensors_camera/
├── schemas/
├── docs/
├── reports/
├── examples/
└── tests/
```

## Pipeline

```text
Raw Dataset
    v
Adapter
    v
Scenario
    v
QA Generation
    v
VLM Inference
    v
Claim Extraction
    v
Ground Truth Comparison
    v
Evaluation Report
```

## Adapters

- `adapters/carla_adapter.py` inspects one local CARLA simulation folder, discovers MP4 videos, RGB frames, JSON logs, semantic segmentation, lidar files, and returns normalized paths, counts, and modalities.
- `adapters/smartroom_adapter.py` inspects one local Smart Room recording folder, discovers camera/video files, optional frames, optional sensors, optional `metadata.json`, and returns normalized paths, counts, and modalities.
- Adapters are lightweight dataset discovery modules. They should not pull raw data from another repository or load large media/sensor payloads into memory.

## Scenario Sources

- `scenarios/carla/` contains CARLA simulation benchmark scenarios.
- `scenarios/smartroom_sensors_camera/` contains Smart Room camera/sensor benchmark scenarios.

## Local Data

- `data/` is the local raw input data area and is ignored by git except for `.gitkeep` placeholders.
- `data/carla/` contains CARLA simulations, for example `data/carla/simulation_45/`.
- `data/smartroom/` contains Smart Room recordings, for example `data/smartroom/recording_001/`.
- Raw videos, frames, lidar, and sensor dumps should stay local and should not be committed.

Example local CARLA input:

```text
data/carla/simulation_45/
├── simulation_45.mp4
├── frames/
├── json/
├── semseg/
└── lidar/
```

## Scripts

- `scripts/` contains scenario creation, VLM execution, claim conversion, validation, and scoring scripts.
- `scripts/run_vlm_on_simulation.py` can run from local CARLA input:

```bash
python3 scripts/run_vlm_on_simulation.py --sim-dir data/carla/simulation_45 --scene-id carla_simulation_45 --input-mode auto --model-path /home/native/internvl3/InternVL/pretrained/InternVL3-9B
```

## Scenario Layout

Each scenario should eventually contain:

- `scenario.json`
- `ground_truth.json`
- `qa_pairs.json`
- `sample_answers.json`
- `frames_sampled/`
- `model_answers/`
- `claims/`
- `reports/`

## Benchmark Taxonomy

GroundingEval scenarios are organized by difficulty level:

- Easy
- Medium
- Hard

Each difficulty contains five reasoning categories:

1. Object Recognition

- object presence
- counting
- attributes

2. Spatial Reasoning

- relative position
- containment
- distance
- lane/crosswalk reasoning

3. Temporal Reasoning

- before/after
- entering/leaving scene
- motion

4. Scene Understanding

- overall interpretation
- environment understanding
- context

5. Safety Reasoning

- hazards
- driving decisions
- risk assessment
- unsupported conclusions

The benchmark should eventually report per-category accuracy in addition to
overall accuracy.
