"""Bet 16 formal test: do detection thresholds order as feel <= judge <= structural?

Detection definitions (fixed here, stated in the report):
  feel       smallest dose where |feel log-ratio shift| exceeds the shuffled
             control's |shift| at ITS matched dose (the vocabulary-noise yardstick).
  judge      smallest dose where Mann-Whitney vs alpha=0 gives p < 0.05 (n=20/cell).
  structural smallest dose where the item-cluster bootstrap 95% CI of the gain beta
             excludes 1.0 (the pre-registered CI rule, D30/prereg rule 5: resample
             ITEMS with replacement, keep edges among distinct sampled items, refit
             baseline mu and dose beta on the induced subgraph).

Run:  uv run --project src python src/bet16_bootstrap.py
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
N_BOOT = 80
rng = np.random.default_rng(20260816)

DOSES = [("b_primary_a1_t1.json", 0.5), ("b_primary_a2_t1.json", 1.0),
         ("b_primary_a3_t1.json", 1.5), ("b_primary_a4_t1.json", 2.0)]


def beta_ci(base_path, dose_path, n_boot=N_BOOT):
    d0, _, tr0, ho0 = load_battery(str(RUNS / base_path))
    dd, _, trd, hod = load_battery(str(RUNS / dose_path))
    n = len(d0["item_ids"])
    betas = []
    for b in range(n_boot):
        # With-replacement item bootstrap (hostile-review fix): an item drawn k
        # times contributes its edges k_i * k_j times, via multiplicity weights
        # implemented as edge repetition.
        draw = rng.integers(0, n, n)
        mult = np.bincount(draw, minlength=n)
        def wsub(edges):
            out = []
            for i, j, p in edges:
                for _ in range(int(mult[i] * mult[j])):
                    out.append((i, j, p))
            return out
        sub0 = wsub(tr0)
        subd = wsub(trd)
        subh = wsub(hod)
        if len(subd) < 50 or len(subh) < 10:
            continue
        ia, ib, pa = edges_to_arrays(sub0)
        mu, s2, _ = thurstonian.fit(ia, ib, pa, n, num_epochs=500, seed=b)
        g = fit_gain(mu, s2, subd, subh)
        betas.append(g["beta"])
    return np.percentile(betas, [2.5, 97.5]), len(betas)


def main():
    print("structural detection: beta cluster-bootstrap CI per dose (excludes 1.0?)")
    struct_fire = None
    for path, u in DOSES:
        (lo, hi), k = beta_ci("gate9b_a0_t1.json", path)
        excl = lo > 1.0 or hi < 1.0
        if excl and struct_fire is None:
            struct_fire = u
        print(f"  {u:+.1f}u: beta CI [{lo:.3f}, {hi:.3f}] ({k} reps) "
              f"{'EXCLUDES 1 — structural change detected' if excl else 'includes 1'}")
    feel = {0.5: 0.776, 1.0: 0.924, 1.5: 0.894, 2.0: 0.829}   # |shift| from R4 feel data
    noise = 0.304                                              # |shuffled shift| at its 0.5u dose
    feel_fire = min((u for u, s in feel.items() if s > noise), default=None)
    judge_fire = 1.0                                           # R5: p=.11 at 0.5u, p=.020 at 1u
    print(f"\nthresholds: feel fires at {feel_fire}u (noise yardstick {noise}), "
          f"judge at {judge_fire}u, structural at {struct_fire}u")
    ok = feel_fire <= judge_fire and judge_fire <= (struct_fire or 99)
    print(f"bet 16 ordering feel<=judge<=structural: {'HIT' if ok else 'MISS'}")


if __name__ == "__main__":
    main()
