# hackhive_control_hub_loc

UR Hackathon 2026H2: joint optimisation of hub placement and closed-ring cable
routing on ResPlan floor plans. Fewest hubs wins; ties go to the lowest
J = L + 5X + O + 0.5B.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`data/sample300.pkl` (300 plans) is committed, so the demos run as-is.
For the full 17,000-plan set, download `ResPlan.pkl` from Kaggle
(kaggle.com/datasets/resplan/resplan) into `archive (1)/` and pass `--full`.

## Layout

- `hubring/plan.py`: load a plan, convert to metres, eligible rooms, fixed node per room, door graph, free space
- `hubring/grid.py`: non-uniform axis-aligned routing grid (Phase 0)
- `hubring/routing.py`: bend-aware Dijkstra over (grid point, heading); `route_table` gives L + 0.5B leg costs between nodes (Phase 1)
- `scripts/phase0_demo.py`, `scripts/phase1_demo.py`: demos; plots go to `out/` (git-ignored)
- `archive (1)/`: ResPlan's own utilities (`resplan_utils.py`) and docs

```bash
.venv/bin/python scripts/phase0_demo.py --random 1
.venv/bin/python scripts/phase1_demo.py --random 1
.venv/bin/python scripts/phase1_demo.py --verify
```

## Next phases

2. List valid groups: connected via doors / open passages, ≤ 100 m² eligible area, ≤ 5 nodes.
3. Best ring per group: try the ≤ 12 node orders with a heading DP over `route_table`; place the hub mid-way along the longest straight run; compute exact X and O.
4. Bitmask DP over rooms: minimise (hub count, total J); tie-break on the most even assigned area.
5. JSON output, our own validator that recomputes H/L/X/O/B from polylines, overlay plot.
