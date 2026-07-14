# Reference Documents

Project context documents for GroundingEval. This folder exists so the core
project docs live in the repo and don't have to be re-shared for context.
Summaries below capture the key points of each document.

## Documents

| File | What it is |
|---|---|
| `PROJECT_OVERVIEW.pdf` | Overall project overview: TeLLMe scene understanding, the role of grounding evaluation, and how the teams/workstreams fit together. |
| `TEAM_3_GROUNDINGEVAL.pdf` | Team 3's charter: build GroundingEval, a claim-level grounding benchmark for multimodal scene understanding, sourced first from CARLA simulations and later Smart Room recordings. |
| `EVAL_QUICKSTART.pdf` | Quickstart for running the evaluation pipeline: environment setup, running the VLM on a simulation, and producing scored outputs. |
| `cityos.pdf` | CityOS paper (Columbia/Rutgers): the urban-sensing platform whose applications motivate why grounded, hallucination-free scene understanding matters. |

## Key context (for anyone or any tool picking this project up)

### The project in one paragraph

GroundingEval evaluates whether multimodal models (VLMs) can describe visual
scenes **without hallucinating** objects, events, actions, or relationships.
Model outputs are decomposed into atomic **claims** (presence, count,
attribute, spatial relation, event) and each claim is scored against
**deterministic ground truth** derived from simulator/sensor state — not from
human annotation. The first data source is CARLA driving simulation; the
second is Smart Room camera/sensor recordings. The repo pipeline is:
adapter → scenario → QA generation → VLM inference → claim extraction →
ground-truth comparison → evaluation report.

### CityOS (why this benchmark exists)

CityOS is an operating system for urban sensing (cameras/sensors at
intersections, parking lots, transit stops). Core idea: a **same-location
policy** — a physical analog of the web's same-origin policy — where sensor
data and computation stay local by default, and broader access goes through
three risk-calibrated API tiers:

- **API 1 (On-Scene):** real-time, ephemeral access; outputs confined to the
  local context (e.g., pedestrian-vehicle collision alerts). Apps run in
  ephemeral containers with a bounded frame window (minEC/maxEC).
- **API 2 (Single-Locality Aggregation):** longitudinal statistics at one
  location released only with differential privacy (e.g., hourly
  pedestrian/vehicle counts for a dashboard).
- **API 3 (Cross-Locality Aggregation):** citywide measurement mediated by
  user devices with per-user privacy budgets (e.g., subway route popularity
  without tracking riders).

Its prototype applications — pedestrian safety alerts, parking availability,
traffic dashboards, trajectory measurement — all depend on **accurate machine
perception of street scenes** (object detection/tracking of pedestrians,
vehicles, cyclists; occupancy of regions like parking spots and crosswalks;
collision-course detection). GroundingEval is the benchmark side of that
stack: it measures whether scene-understanding models can be trusted to
report what actually happened, which matters for safety-critical, in-context
decisions (API 1) and for the correctness of aggregated statistics (API 2/3).

### The benchmark design (agreed so far)

- **Difficulty is two-dimensional** (team whiteboard idea): *scene difficulty*
  (how hard the pixels are: actor count, occlusion, fog, entries/exits) ×
  *question difficulty* (how hard the reasoning is: direct lookup →
  time/attribute-constrained lookup → compositional/causal/counterfactual).
  Each category reports a 3×3 accuracy grid; the gradient direction
  distinguishes perception-limited vs. reasoning-limited vs.
  hallucination-limited models.
- **Claim types:** counting/occupancy, relation, attribute/state,
  temporal/event, adversarial/ambiguity.
- **Scenario taxonomy** (full spec in
  `GroundingEval_CARLA_Scenario_Taxonomy.docx`, shared separately): 10
  categories — Census/Counting, Attribute Binding, Motion & State Change,
  Regions & Spatial Relations, Temporal Order & Event Chains, Identity
  Persistence, Contact vs. Near-Miss, Signal Grounding, Existence vs.
  Visibility, Null Events & Expectation Traps. v1 recommendation: 7 Tier-1
  categories × 3 scene tiers × 5 seeds ≈ 105 short simulations, one seed per
  tier being a negative/trap variant.
- **Ground-truth rules:** every claim must be derivable deterministically
  from logs/sensors (world-state per tick, instance segmentation for
  visibility, collision sensors, map polygons, traffic/vehicle light state).
  Counterfactual questions are grounded by a **twin simulation** (same seed,
  one scripted intervention, short horizon).
- **Difficulty labels are auditable:** each simulation stores its
  difficulty-dimension values computed from its own logs (occlusion duration,
  event gaps, actor counts), so Easy/Medium/Hard is checkable, not asserted.

### Current repo state (as of 2026-07)

- `adapters/carla_adapter.py` discovers modalities in a local CARLA
  simulation folder (frames, video, json logs, semseg, lidar).
- `data/carla/simulation_45` and `simulation_50` are "scenario 4" crosswalk
  recordings (Town10HD): per-tick JSON logs (crosswalk ROI occupancy from
  semseg pixels, ego speed, fog, collision counts) + ground-truth claims
  (presence, stopped/max-speed attributes, crosswalk occupancy, enter/exit).
- Known gaps to close when recording new scenarios: `ego_x/y/yaw` and
  `frame_cam2` are logged but always null; no instance segmentation; count
  fields never written (the GT metadata says so explicitly); no lidar files
  despite adapter support; crosswalk occupancy uses a hardcoded pixel ROI
  instead of map polygons.
- `scripts/run_vlm_on_simulation.py` runs InternVL3-9B on a simulation
  (video-first input); `generate_qa_pairs.py` and `extract_claims.py` are
  placeholders that the two-axis design above is meant to specify.
