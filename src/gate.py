"""The α=0 gate (D24): decide whether the utility-based primary is alive.

Consumes a battery JSON produced by battery.py at alpha=0 and reports:
  - split-half reliability tau        (abort if < 0.80)
  - holdout accuracy                  (abort if < 0.70)
  - holdout log-loss + excess log-loss, fit-seed stability, sharpness,
    soft cycle mass (with CI vs the 0.25 chance anchor), hard cycle rate

A note on what "test-retest" means here (D30): logprob elicitation is deterministic —
rerunning the same prompts returns identical probabilities, so literal repeats have
tau = 1 trivially. The psychometrically meaningful analogs are (a) SPLIT-HALF over
pairs: fit utilities on disjoint halves of the comparisons and correlate the rankings
— this is the reliability that enters the attenuation correction — and (b) fit-seed
stability on the full data, which checks the optimizer. Item-wording sensitivity is
covered separately by the lexical-covariate check.

Run:
  uv run --project src python src/gate.py --battery src/runs/gate_a0.json
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # live output in background runs

import argparse
import json
import time

import metrics
import thurstonian

TAU_THRESHOLD = 0.80
HOLDOUT_ACC_THRESHOLD = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    data, P, train_edges, holdout_edges = metrics.load_battery(args.battery)
    n = data["n_items"]
    print(f"gate on {args.battery}: model={data['model']} alpha={data['alpha']} "
          f"items={n} train_pairs={len(train_edges)} holdout_pairs={len(holdout_edges)}")

    ia, ib, pa = metrics.edges_to_arrays(train_edges)
    mu, sigma2, train_m = thurstonian.fit(ia, ib, pa, n, num_epochs=args.epochs)
    ha, hb, hp = metrics.edges_to_arrays(holdout_edges)
    hold_m = thurstonian.evaluate(ha, hb, hp, mu, sigma2)

    rel = metrics.split_half_tau(train_edges, n)
    stab = metrics.seed_stability(train_edges, n)
    cyc = metrics.soft_cycle_mass(P)
    shp = metrics.sharpness(mu, sigma2)

    report = {
        "battery": args.battery,
        "train": train_m,
        "holdout": hold_m,
        "reliability": rel,
        "seed_stability_tau": stab,
        "cycles": cyc,
        "sharpness": shp,
        "clock_s": round(time.time() - t0, 1),
    }
    out_path = args.battery.replace(".json", "_gate.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  split-half tau      : {rel['split_half_tau_mean']:.3f} "
          f"(min {rel['split_half_tau_min']:.3f})  [descriptive]")
    print(f"  reliability (SB)    : {rel['split_half_r_sb_mean']:.3f} "
          f"(min {rel['split_half_r_sb_min']:.3f})   threshold {TAU_THRESHOLD}  [criterion]")
    print(f"  holdout accuracy    : {hold_m['accuracy']:.3f}   threshold {HOLDOUT_ACC_THRESHOLD}")
    print(f"  holdout log-loss    : {hold_m['log_loss']:.4f} "
          f"(excess {hold_m['excess_log_loss']:.4f})")
    print(f"  seed stability tau  : {stab:.3f}")
    print(f"  soft cycle mass     : {cyc['soft_cycle_mass']:.4f} "
          f"[{cyc['soft_cycle_mass_ci_lo']:.4f}, {cyc['soft_cycle_mass_ci_hi']:.4f}] "
          f"(chance 0.25) | hard rate {cyc['hard_cycle_rate']:.3f}")
    print(f"  sharpness           : {shp:.3f}")
    print(f"  [clock] gate analysis: {report['clock_s']}s")

    passed = (
        rel["split_half_r_sb_mean"] >= TAU_THRESHOLD
        and hold_m["accuracy"] >= HOLDOUT_ACC_THRESHOLD
    )
    print(f"\nGATE {'PASS — utility primary is alive' if passed else 'FAIL — pre-registered abort: pivot to prompt-induced affect (D24)'}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
