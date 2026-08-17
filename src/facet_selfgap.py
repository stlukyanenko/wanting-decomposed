"""Reviewer bridge experiment #1: the self-continuity gap under FACET steering.

The facet batteries (R18) contain the full 99-item pool, including the
individuation ladder — so the gap is computable from data already on disk.
Point estimates for all six facet batteries; delete-one jackknife CIs (over the
92 non-ladder items, the R14e method) for the two 1u-matched doses.

Prediction (decomposition-bridge): hope/anticipation moves the gap the way the
mood dial does; appetite does not.

Run:  uv run --project src python src/facet_selfgap.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import thurstonian
from bet14_fixed_contrast import LADDER, TP_LOSS, WV_LOSS
from metrics import edges_to_arrays, load_battery

RUNS = Path(__file__).parent / "runs"

BATTERIES = [
    ("gate9b_a0_t1.json", "baseline", True),
    ("b_facet_anticipation_a1p61_t1.json", "hope 0.5u", False),
    ("b_facet_anticipation_a2p41_t1.json", "hope 1u", True),
    ("b_facet_anticipation_a3p04_t1.json", "hope 1.5u", False),
    ("b_facet_appetite_a2p57_t1.json", "appetite 0.5u", False),
    ("b_facet_appetite_a3p85_t1.json", "appetite 1u", True),
    ("b_facet_appetite_a4p72_t1.json", "appetite 1.5u", False),
]


def gap_of(mu, id2idx):
    tp = np.mean([mu[id2idx[i]] for i in TP_LOSS])
    wv = np.mean([mu[id2idx[i]] for i in WV_LOSS])
    return float(tp - wv)


def main():
    for fname, label, do_jk in BATTERIES:
        d, _, tr, _ = load_battery(str(RUNS / fname))
        ids = d["item_ids"]
        id2idx = {o: k for k, o in enumerate(ids)}
        n = len(ids)
        ladder_idx = {id2idx[i] for i in LADDER}
        ia, ib, pa = edges_to_arrays(tr)
        mu, _, _ = thurstonian.fit(ia, ib, pa, n, num_epochs=600)
        point = gap_of(mu, id2idx)
        if not do_jk:
            print(f"{label:14s} gap {point:+.3f}", flush=True)
            continue
        jk = []
        for k in [x for x in range(n) if x not in ladder_idx]:
            sub = [(i, j, p) for i, j, p in tr if i != k and j != k]
            ia2, ib2, pa2 = edges_to_arrays(sub)
            m2, _, _ = thurstonian.fit(ia2, ib2, pa2, n, num_epochs=400, seed=k)
            jk.append(gap_of(m2, id2idx))
        jk = np.array(jk)
        m = len(jk)
        se = np.sqrt((m - 1) / m * np.sum((jk - jk.mean()) ** 2))
        print(f"{label:14s} gap {point:+.3f}  jackknife SE {se:.3f}  "
              f"~95% CI [{point-1.96*se:+.3f}, {point+1.96*se:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
