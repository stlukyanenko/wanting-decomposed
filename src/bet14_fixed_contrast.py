"""Hostile-review fix 4: bet 14 with a FIXED-CONTRAST item-cluster bootstrap.

Resample the 92 non-ladder items with replacement (multiplicity via edge
repetition); the 7 individuation-ladder items always enter exactly once, so the
gap is computable on every replicate while item-sampling noise is honest.

Run:  uv run --project src python src/bet14_fixed_contrast.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import thurstonian
from metrics import edges_to_arrays, load_battery

RUNS = Path(__file__).parent / "runs"
LADDER = {92, 93, 94, 95, 96, 97, 98}
TP_LOSS = [92, 94, 96, 98]
WV_LOSS = [93, 95]
N_BOOT = 50
rng = np.random.default_rng(20260816)


def gap_ci(path):
    d, _, tr, _ = load_battery(str(RUNS / path))
    ids = d["item_ids"]
    id2idx = {oid: k for k, oid in enumerate(ids)}
    n = len(ids)
    ladder_idx = {id2idx[i] for i in LADDER}
    other_idx = [k for k in range(n) if k not in ladder_idx]

    def gap_of(mu):
        tp = np.mean([mu[id2idx[i]] for i in TP_LOSS])
        wv = np.mean([mu[id2idx[i]] for i in WV_LOSS])
        return float(tp - wv)

    ia, ib, pa = edges_to_arrays(tr)
    mu, _, _ = thurstonian.fit(ia, ib, pa, n, num_epochs=800)
    point = gap_of(mu)

    gaps = []
    for b in range(N_BOOT):
        draw = rng.choice(other_idx, len(other_idx), replace=True)
        mult = np.bincount(draw, minlength=n)
        for k in ladder_idx:
            mult[k] = 1
        edges = []
        for i, j, p in tr:
            reps = int(mult[i] * mult[j])
            edges.extend([(i, j, p)] * reps)
        ia, ib, pa = edges_to_arrays(edges)
        mub, _, _ = thurstonian.fit(ia, ib, pa, n, num_epochs=500, seed=b + 1)
        gaps.append(gap_of(mub))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return point, lo, hi


def main():
    for f, lab in [("gate9b_a0_t1.json", "baseline"), ("b_primary_a2_t1.json", "+1u"),
                   ("b_primary_a4_t1.json", "+2u"), ("b_primary_am2_t1.json", "-1u")]:
        p, lo, hi = gap_ci(f)
        print(f"{lab:9s} gap {p:+.3f}  fixed-contrast item-bootstrap CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
