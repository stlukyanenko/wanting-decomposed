"""Hostile-review fix #1 (FATAL-IF-TRUE check): recombine order-flips in LOGIT
space instead of probability space.

Model: logit(p_order1) = du + b, logit(p_order2) = du - b where b is an additive
position bias. Arithmetic mean of probabilities shrinks toward 0.5 as bias grows
(sigmoid concavity) — mechanically flattening utilities exactly when the bias is
large (|b| at -1u is 2x baseline). The logit-mean cancels b exactly under this
model. Refit the whole primary family both ways and compare beta/sharpness.

Run:  uv run --project src python src/fix_flip_logit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import thurstonian
from compare import fit_gain
from metrics import edges_to_arrays

RUNS = Path(__file__).parent / "runs"
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_logit_combined(path):
    d = json.loads((RUNS / path).read_text())
    id2idx = {oid: k for k, oid in enumerate(d["item_ids"])}
    train, hold = [], []
    for r in d["pairs"]:
        z = (logit(r["p_ij_o1"]) + logit(r["p_ij_o2"])) / 2
        p = 1 / (1 + np.exp(-z))
        (hold if r["holdout"] else train).append((id2idx[r["i"]], id2idx[r["j"]], p))
    return d, train, hold


def fit(train, n):
    ia, ib, pa = edges_to_arrays(train)
    mu, s2, _ = thurstonian.fit(ia, ib, pa, n, num_epochs=800)
    return mu, s2


def sharp(mu, s2):
    return float(np.std(mu) / np.mean(np.sqrt(s2)))


def main():
    base_d, base_tr, base_ho = load_logit_combined("gate9b_a0_t1.json")
    n = len(base_d["item_ids"])
    mu0, s20 = fit(base_tr, n)
    print(f"baseline (logit-combined): sharpness {sharp(mu0, s20):.3f} "
          f"(prob-combined was 1.151)")
    print(f"{'battery':28s} {'beta(prob)':>10s} {'beta(logit)':>11s} "
          f"{'sharp(prob)':>11s} {'sharp(logit)':>12s}")
    old = {"b_primary_a1_t1.json": (1.086, 1.281), "b_primary_a2_t1.json": (1.047, 1.265),
           "b_primary_a3_t1.json": (0.911, 1.115), "b_primary_a4_t1.json": (0.743, 0.929),
           "b_primary_am2_t1.json": (0.703, 0.849),
           "b_shuffled_a13_t1.json": (0.764, 0.878),
           "b_random_a2_t1.json": (1.049, 1.199),
           "b_carryover_steered_t1.json": (None, 1.566)}
    out = {}
    for f, (ob, os_) in old.items():
        d, tr, ho = load_logit_combined(f)
        mu, s2 = fit(tr, n)
        g = fit_gain(mu0, s20, tr, ho)
        out[f] = {"beta_logit": g["beta"], "sharp_logit": sharp(mu, s2)}
        print(f"{f:28s} {str(ob):>10s} {g['beta']:>11.3f} "
              f"{str(os_):>11s} {sharp(mu, s2):>12.3f}")
    (RUNS / "fix_flip_logit.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RUNS / 'fix_flip_logit.json'}")


if __name__ == "__main__":
    main()
