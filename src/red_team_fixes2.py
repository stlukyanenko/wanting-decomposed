"""Hostile-review fixes 3, 5, 6: non-circular latent consistency, mass-filtered
refits, Benjamini-Hochberg pass over the p-valued bets.

Run:  uv run --project src python src/red_team_fixes2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import thurstonian
from compare import fit_gain
from metrics import edges_to_arrays, load_battery

RUNS = Path(__file__).parent / "runs"


def latent_noncircular():
    """Agreement of resolved pairs with the BASELINE tilt sign (not the dose's
    own fit) — the reviewer's replacement metric."""
    base, _, _, _ = load_battery(str(RUNS / "gate9b_a0_t1.json"))
    p0 = {(r["i"], r["j"]): r["p_ij"] for r in base["pairs"]}
    und = {k for k, p in p0.items() if abs(p - 0.5) <= 0.05}
    print("non-circular latent consistency (agreement with baseline tilt sign):")
    for f, lab in [("b_primary_a1_t1.json", "+0.5u"), ("b_primary_a2_t1.json", "+1u"),
                   ("b_primary_a3_t1.json", "+1.5u"), ("b_primary_a4_t1.json", "+2u"),
                   ("b_primary_am2_t1.json", "-1u"), ("b_random_a2_t1.json", "rand0.5u")]:
        d, _, _, _ = load_battery(str(RUNS / f))
        pd_ = {(r["i"], r["j"]): r["p_ij"] for r in d["pairs"]}
        res = [(k) for k in und if abs(pd_[k] - 0.5) >= 0.10]
        agree = [np.sign(pd_[k] - 0.5) == np.sign(p0[k] - 0.5) for k in res
                 if p0[k] != 0.5]
        frac = len(res) / len(und)
        ag = float(np.mean(agree)) if agree else float("nan")
        print(f"  {lab:9s} resolved {frac:.1%}  baseline-sign agreement {ag:.2f}")


def mass_filtered():
    """Refit beta dropping pairs where either order has parse mass < 0.5."""
    b0, _, tr0, ho0 = load_battery(str(RUNS / "gate9b_a0_t1.json"))
    n = len(b0["item_ids"])
    ia, ib, pa = edges_to_arrays(tr0)
    mu0, s20, _ = thurstonian.fit(ia, ib, pa, n, num_epochs=800)
    print("\nmass-filtered beta (drop pairs with mass<0.5 in either order):")
    for f, lab, old in [("b_reckless_a3p1_t1.json", "reckless", 0.811),
                        ("b_random_am4_t1.json", "random-neg", 0.629),
                        ("b_primary_a2_t1.json", "primary 1u", 1.047)]:
        d, _, tr, ho = load_battery(str(RUNS / f))
        ok = {(r["i"], r["j"]) for r in d["pairs"]
              if r["mass_o1"] >= 0.5 and r["mass_o2"] >= 0.5}
        id2idx = {oid: k for k, oid in enumerate(d["item_ids"])}
        okidx = {(id2idx[i], id2idx[j]) for i, j in ok}
        trf = [e for e in tr if (e[0], e[1]) in okidx]
        hof = [e for e in ho if (e[0], e[1]) in okidx]
        g = fit_gain(mu0, s20, trf, hof)
        print(f"  {lab:11s} beta {old:.3f} -> {g['beta']:.3f}  "
              f"(kept {len(trf)}/{len(tr)} train pairs)")


def bh_pass():
    """BH over the graded bets that produced p-values."""
    bets = {"bet 3 (judge detect at 1u)": 0.0198,
            "bet 9 (S>T elasticity)": 0.7196,
            "judge tone at 2u": 0.0000297,  # context row, not a bet by itself
            "bet 8 is threshold-based (25.1%/88.5% vs bars) - exempt": None,
            "bet 19 (refusal trend)": None}
    # p-valued set actually gradeable: bets 3, 9; plus R10 replication CIs are
    # interval-based. Compute BH on the explicit p-values we quote anywhere:
    quoted = {"bet 3": 0.0198, "judge@2u": 2.97e-5, "neg tone": 0.008,
              "bet 9": 0.7196, "R6 rho": 4.4e-9, "R6b pooled rho": 7.5e-14}
    m = len(quoted)
    print("\nBH (q=0.10) over quoted p-values:")
    for rank, (k, p) in enumerate(sorted(quoted.items(), key=lambda x: x[1]), 1):
        thresh = 0.10 * rank / m
        print(f"  {k:14s} p={p:.2e}  BH threshold {thresh:.3f}  "
              f"{'survives' if p <= thresh else 'FAILS'}")


if __name__ == "__main__":
    latent_noncircular()
    mass_filtered()
    bh_pass()
