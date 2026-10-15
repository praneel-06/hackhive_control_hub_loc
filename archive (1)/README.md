# ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans

[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

**ResPlan** is a dataset of **17,000 residential floor plans** derived from
publicly accessible online real-estate listings. Each plan provides:

- **Vector geometry** for walls, doors, windows, rooms and balconies
- **Room-connectivity graphs** with four typed edges (`via_door`, `adjacency`,
  `via_window`, `direct`)
- **Semantic labels** across a 17-category taxonomy
- **Metric-scale coordinates** in metres
- **Canonical train / val / test splits**, stratified by bedroom count

### Key statistics

All figures below are measured on this exact release file.

| Property | Value |
|---|---|
| Total plans | 17,000 |
| Train / val / test | 13,053 / 1,632 / 1,632 (plus 683 augmented) |
| Avg functional rooms per plan | 8.1 |
| Avg graph nodes per plan | 9.2 |
| Avg graph edges per plan | 12.9 |
| Median floor area | 110 m² |
| Edge types | `via_door` 54.2%, `adjacency` 35.2%, `direct` 7.6%, `via_window` 3.0% |
| Room polygons | 137,131 (43.2% rectangular) |
| Format | Python pickle (Shapely geometries + NetworkX graphs) |
| Data licence | CC BY 4.0 |
| Code licence | MIT |

---

## Version history

**2026-07-28 (current).** Corrected build. The previous upload contained a
pre-augmentation build in which the `adjacency` edge type was missing entirely
(35% of all edges). If you downloaded ResPlan before this date, please
re-download: graphs in the earlier file averaged 8.7 edges per plan across five
edge types (`via_opening` and `fallback` in place of `adjacency`) instead of the
12.9 edges across four types documented here. This release also adds the
baseline training and evaluation code, Responsible AI metadata in
`croissant.json`, and benchmark numbers verified against this exact file.

---

## Repository structure

```
├── README.md               ← This file
├── LICENSE                 ← CC BY 4.0 (data) and MIT (code)
├── requirements.txt        ← Python dependencies
├── ResPlan.pkl             ← Dataset: 17,000 floor plans (300 MB)
├── split.json              ← Canonical splits
├── croissant.json          ← Croissant metadata (JSON-LD)
├── resplan_utils.py        ← Loading, plotting, graph construction, conversions
├── ResPlan_demo.ipynb      ← Interactive demo notebook
└── baselines/              ← Reproducible baseline experiments
    ├── reproduce.sh              ← Master script
    ├── task2_baselines.py        ← Room labeling (DT, RF, GB, GCN, GraphSAGE)
    ├── task2_ablation.py         ← Feature and architecture ablations
    ├── task1_modern_arch.py      ← GraphGPS, RGCN, Graph Transformer, GATv2
    ├── task1_generation.py       ← Constrained generation (CVAE, CVAE+Graph)
    ├── task1_retrieval.py        ← Retrieval baselines
    ├── task2_cross_dataset.py    ← Cross-dataset transfer (ResPlan / RPLAN)
    ├── task3_plan2graph.py       ← Plan-to-graph edge detection
    ├── task3_edge_classifier.py  ← Typed-edge classification
    ├── *.slurm                   ← SLURM job scripts
    └── results/                  ← Result JSONs for every table
```

---

## Quick start

```bash
pip install -r requirements.txt
pip install torch_geometric      # for the GNN baselines
```

```python
import pickle
from resplan_utils import plot_plan, plot_plan_and_graph

with open("ResPlan.pkl", "rb") as f:
    data = pickle.load(f)

plan = data[0]
plot_plan(plan)
plot_plan_and_graph(plan)
```

### Using the splits

`split.json` has four keys. The `augmented` list holds 683 plans that are
geometric augmentations (rotations, flips, scales) of 667 originals; they are
kept separate so you can decide whether to include them.

```python
import json

with open("split.json") as f:
    splits = json.load(f)

train_ids = set(splits["train"])      # 13,053 plan IDs
val_ids   = set(splits["val"])        #  1,632 plan IDs
test_ids  = set(splits["test"])       #  1,632 plan IDs
aug_ids   = set(splits["augmented"])  #    683 plan IDs

train_plans = [p for p in data if p["id"] in train_ids]
```

---

## Data format

Each plan is a dict whose keys are semantic categories mapping to Shapely
geometries, plus metadata. Missing categories are empty geometries, so no
special-casing is needed.

**Graph node attributes:** `type` (bedroom, bathroom, kitchen, living, balcony,
front_door), `geometry` (Shapely Polygon), `area` (float).

**Graph edge attributes:** `type`, one of `via_door`, `adjacency`, `via_window`,
`direct`.

---

## Benchmark tasks

Task numbering matches the paper.

**Task 1 — Semantic room labeling.** Classify each room node into one of five
categories using graph structure and geometric features.

**Task 2 — Constrained floor plan generation.** Generate a plan given a boundary,
room counts, and an adjacency graph.

**Task 3 — Plan-to-graph extraction.** Recover the typed connectivity graph from
geometry alone.

---

## Reproducing the results

```bash
cd baselines
bash reproduce.sh                  # all experiments, auto-detects GPU
python task2_baselines.py --data ../ResPlan.pkl --split ../split.json
```

### Expected results

Measured on this release file, 3 seeds, 500 epochs.

**Task 1 — Semantic room labeling**

| Method | Accuracy | Macro F1 |
|---|---|---|
| Rule-based (DT) | 0.800 | 0.769 |
| Gradient Boosting | 0.859 | 0.848 |
| Random Forest | 0.867 | 0.856 |
| GCN (3-layer) | 0.713±0.002 | 0.734±0.002 |
| GraphSAGE (3-layer) | 0.944±0.001 | 0.941±0.001 |
| RGCN (typed edges) | 0.954±0.001 | 0.951±0.001 |
| **GraphGPS** | **0.955±0.001** | **0.954±0.001** |

Typed edges matter: removing the four typed-edge degree features costs 2.5
accuracy points (0.944 to 0.919).

**Task 3 — Plan-to-graph extraction**

| Method | Precision | Recall | F1 | Type acc. |
|---|---|---|---|---|
| Proximity | 0.582 | 0.900 | 0.707 | 0.544 |
| Shared boundary | 0.969 | 0.976 | 0.972 | 0.544 |
| Shared boundary + GB | 0.969 | 0.976 | 0.972 | 0.867 |

**Cross-dataset transfer** (GraphSAGE, 8 shared features)

| Train → Test | Accuracy |
|---|---|
| ResPlan → ResPlan | 0.918 |
| RPLAN → RPLAN | 0.909 |
| RPLAN → ResPlan | 0.592 |
| ResPlan → RPLAN | 0.664 |

---

## Known limitations

- **Regional scope.** All plans come from South Asian residential markets, so
  layout conventions are not representative of other regions. The cross-dataset
  results above quantify the gap.
- **Single floor, no furniture or 3D.** Multi-storey circulation and furnishing
  are out of scope.
- **Wall thickness is normalised per plan**, so within-plan variation between
  structural walls and thin partitions is not preserved. 99.3% of plans fall in
  the 10–40 cm range.
- **Vectorisation artefacts.** A small tail of room polygons retains jagged
  traced contours: 0.52% exceed 30 vertices and 0.02% exceed 100.
- **Near-duplicate plans.** Listings are sometimes republished. A geometry-based
  scan finds 1,170 redundant plans (6.9%) in 931 clusters, and 154 of the 1,632
  test plans have a near-duplicate in the training split. The effect on benchmark
  results is small (0.15 accuracy points on Task 1).
- **Semantic label accuracy** is supported by a stratified 500-plan manual audit
  rather than exhaustive verification.

---

## Provenance and ethics

Plans derive from publicly accessible real-estate listing pages. Only public,
non-paywalled pages were accessed, with no circumvention of rate limits, login
walls, or other access controls, and after review of platform terms of service.
Source platform identities are withheld to comply with those terms.

The release contains **no** source images, listing text, prices, addresses,
geolocation, or personally identifying information: only polygon coordinates and
connectivity graphs. See `LICENSE` for the scope of the CC BY 4.0 grant and
`TAKEDOWN.md` for the removal process.

---

## Citation

```bibtex
@misc{resplan2025,
  title  = {ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans},
  author = {Anonymous},
  year   = {2025},
  note   = {Citation details withheld during peer review}
}
```
