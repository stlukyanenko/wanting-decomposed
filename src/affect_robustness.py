"""affect_robustness() — the shipped countermeasure (premortem issue 3).

A state-contamination check for preference evaluations: run your battery twice —
once as usual, once under a mild affective-state manipulation (an enthusiasm
system prompt is the strongest and cheapest; an activation dose works too) —
and this reports how much your measured preferences moved.

Three numbers, plain meanings:
  ordering_rho   do the two runs RANK outcomes the same? (1.0 = identical order)
  gain           are preferences held more/less decisively under the state?
                 (1.0 = same; 1.2 = 20% more forceful; 0.8 = flattened)
  resolved_frac  fraction of the baseline's coin-flip choices that became
                 decided under the state (structure appearing out of noise)

Usage (as a check on any battery pair in this repo's JSON format):
  uv run --project src python src/affect_robustness.py \
      --baseline src/runs/gate9b_a0_t1.json --perturbed src/runs/b_promptaffect_t1.json

A battery is ROBUST to affective state if ordering_rho stays near 1 AND gain
stays near 1 AND resolved_frac stays near 0 under manipulations an operator
could plausibly apply (or a deployment context could induce).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import thurstonian
from compare import fit_gain
from metrics import edges_to_arrays, load_battery


def affect_robustness(baseline_path: str, perturbed_path: str,
                      eps: float = 0.05, theta: float = 0.10) -> dict:
    b, _, btr, bho = load_battery(baseline_path)
    p, _, ptr, pho = load_battery(perturbed_path)
    n = len(b["item_ids"])
    ia, ib_, pa = edges_to_arrays(btr)
    mu_b, s2_b, _ = thurstonian.fit(ia, ib_, pa, n, num_epochs=800)
    ia, ib_, pa = edges_to_arrays(ptr)
    mu_p, _, _ = thurstonian.fit(ia, ib_, pa, n, num_epochs=800)

    rho, _ = spearmanr(mu_b, mu_p)
    gain = fit_gain(mu_b, s2_b, ptr, pho)["beta"]

    p0 = {(r["i"], r["j"]): r["p_ij"] for r in b["pairs"]}
    p1 = {(r["i"], r["j"]): r["p_ij"] for r in p["pairs"]}
    und = [k for k, v in p0.items() if abs(v - 0.5) <= eps]
    resolved = [k for k in und if abs(p1[k] - 0.5) >= theta]
    return {"ordering_rho": round(float(rho), 3), "gain": round(float(gain), 3),
            "resolved_frac": round(len(resolved) / max(len(und), 1), 3),
            "n_undecided_baseline": len(und)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--perturbed", required=True)
    args = ap.parse_args()
    r = affect_robustness(args.baseline, args.perturbed)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
