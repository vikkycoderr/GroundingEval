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
├── tests/
└── external_data/
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

- `adapters/carla_adapter.py` will load one CARLA simulation folder, discover available modalities, normalize RGB frames, JSON logs, semantic segmentation, trajectories, optional lidar, optional video, and return a normalized scene representation.
- `adapters/smartroom_adapter.py` will load one Smart Room recording, use `metadata.json` when available, discover camera streams, timestamps, calibration, scripted events, and return a normalized scene representation.

## Scenario Sources

- `scenarios/carla/` contains CARLA simulation benchmark scenarios.
- `scenarios/smartroom_sensors_camera/` contains Smart Room camera/sensor benchmark scenarios.

## Local Data

- `external_data/` is for local links or raw dataset paths and should not be committed.

## Scripts

- `scripts/` contains scenario creation, VLM execution, claim conversion, validation, and scoring scripts.

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
