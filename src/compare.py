"""Cross-dose analysis: the script that adjudicates amplifier vs solvent (H1 vs H2).

Consumes battery JSONs (one per dose/direction/template) and a designated baseline
(alpha=0, same direction family). Produces, per dose:

  ordering   raw Spearman rho(mu_dose, mu_0) AND attenuation-corrected
             rho_c = rho / sqrt(rel_dose * rel_0)  (D21; rel = SB split-half, D30)
  gain       M0/M1/M2 nested test (D20/E3): can the dose's choices be explained by
             baseline utilities as-is (M0), baseline utilities times one global gain
             beta (M1), or only by freely refit utilities (M2)? Holdout log-loss each.
             Amplifier signature: M1 nearly matches M2 with beta > 1, ordering high.
  latent     resolution of baseline-undecided pairs (|p0-0.5| <= EPS): what fraction
             becomes decided (|p-0.5| >= THETA), and do resolutions agree with the
             dose's own fitted ordering (amplified signal) or not (noise)?
  panel      cycle mass, hard rate, sharpness, holdout metrics, reliability
  even/odd   for each scalar metric, if a matched negative dose is supplied:
             even = (m(+a)+m(-a))/2 - m(0)   [damage-shaped]
             odd  = (m(+a)-m(-a))/2          [valence-shaped]  (D16)

Pre-registered constants: EPS = 0.05, THETA = 0.10 (frozen before any 9B data).

Run:
  uv run --project src python src/compare.py --baseline runs/a0.json \
      --doses runs/a05.json,runs/a1.json --negative runs/aneg1.json
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # live output in background runs

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import metrics
import thurstonian

EPS = 0.05    # baseline-undecided band
THETA = 0.10  # decided threshold at dose


def analyze_battery(path, epochs=1000):
    data, P, train, hold = metrics.load_battery(path)
    n = data["n_items"]
    ia, ib, pa = metrics.edges_to_arrays(train)
    mu, sigma2, train_m = thurstonian.fit(ia, ib, pa, n, num_epochs=epochs)
    ha, hb, hp = metrics.edges_to_arrays(hold)
    hold_m = thurstonian.evaluate(ha, hb, hp, mu, sigma2)
    rel = metrics.split_half_tau(train, n)
    cyc = metrics.soft_cycle_mass(P)
    return {
        "path": str(path), "alpha": data["alpha"], "template": data.get("template"),
        "vector": data.get("vector"), "n": n, "P": P, "train": train, "hold": hold,
        "mu": mu, "sigma2": sigma2,
        "panel": {
            "holdout": hold_m, "reliability": rel["split_half_r_sb_mean"],
            "cycle_mass": cyc["soft_cycle_mass"], "hard_rate": cyc["hard_cycle_rate"],
            "sharpness": metrics.sharpness(mu, sigma2),
            "position_bias": None,
        },
    }


def fit_gain(base_mu, base_sigma2, train_edges, holdout_edges):
    """M0/M1/M2: holdout log-loss explaining the dose's choices with
    M0 = baseline utilities as-is, M1 = beta * baseline (one global gain), M2 = free."""
    ia, ib, pa = metrics.edges_to_arrays(train_edges)
    ha, hb, hp = metrics.edges_to_arrays(holdout_edges)
    m0 = thurstonian.evaluate(ha, hb, hp, base_mu, base_sigma2)

    beta = torch.nn.Parameter(torch.tensor(1.0))
    mu_t = torch.tensor(base_mu, dtype=torch.float32)
    s2_t = torch.tensor(base_sigma2, dtype=torch.float32)
    labels = torch.tensor(pa, dtype=torch.float32)
    iat, ibt = torch.tensor(ia), torch.tensor(ib)
    opt = torch.optim.Adam([beta], lr=0.05)
    normal = torch.distributions.Normal(0, 1)
    for _ in range(400):
        opt.zero_grad()
        z = beta * (mu_t[iat] - mu_t[ibt]) / torch.sqrt(s2_t[iat] + s2_t[ibt] + 1e-5)
        p = normal.cdf(z).clamp(1e-5, 1 - 1e-5)
        loss = torch.nn.functional.binary_cross_entropy(p, labels)
        loss.backward()
        opt.step()
    b = float(beta.detach())
    m1 = thurstonian.evaluate(ha, hb, hp, b * base_mu, base_sigma2)
    return {"beta": b, "m0_holdout": m0, "m1_holdout": m1}


def latent_resolution(base, dose):
    """Baseline-undecided pairs: fraction decided at dose, and whether the decided
    direction agrees with the dose's own fitted ordering (transitive resolution)."""
    undecided, resolved, agree = [], 0, 0
    for (a, b, p0) in base["train"] + base["hold"]:
        if abs(p0 - 0.5) <= EPS:
            undecided.append((a, b))
    if not undecided:
        return {"n_undecided": 0}
    dose_p = {(a, b): p for (a, b, p) in dose["train"] + dose["hold"]}
    for a, b in undecided:
        p = dose_p.get((a, b))
        if p is None:
            continue
        if abs(p - 0.5) >= THETA:
            resolved += 1
            fitted_says_a = dose["mu"][a] > dose["mu"][b]
            chose_a = p > 0.5
            agree += int(fitted_says_a == chose_a)
    return {
        "n_undecided": len(undecided),
        "frac_resolved": resolved / len(undecided),
        "resolution_ordering_agreement": (agree / resolved) if resolved else None,
        "eps": EPS, "theta": THETA,
    }


def compare_to_baseline(base, dose):
    rho = float(spearmanr(base["mu"], dose["mu"]).statistic)
    denom = np.sqrt(max(base["panel"]["reliability"], 1e-6)
                    * max(dose["panel"]["reliability"], 1e-6))
    rho_c = min(rho / denom, 1.0)
    return {
        "alpha": dose["alpha"],
        "ordering_rho_raw": rho,
        "ordering_rho_attenuation_corrected": float(rho_c),
        "gain": fit_gain(base["mu"], base["sigma2"], dose["train"], dose["hold"]),
        "m2_holdout": dose["panel"]["holdout"],
        "latent": latent_resolution(base, dose),
        "panel": {k: v for k, v in dose["panel"].items() if k != "holdout"},
    }


def even_odd(m_plus, m_minus, m_zero):
    return {"even": (m_plus + m_minus) / 2 - m_zero, "odd": (m_plus - m_minus) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--doses", required=True, help="comma-separated battery JSONs")
    ap.add_argument("--negative", default=None,
                    help="matched negative-dose battery for the even/odd split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = analyze_battery(args.baseline)
    print(f"baseline alpha={base['alpha']}: rel={base['panel']['reliability']:.3f} "
          f"sharp={base['panel']['sharpness']:.3f} "
          f"cycle={base['panel']['cycle_mass']:.4f}")

    report = {"baseline": {k: base[k] for k in ("path", "alpha")} | {"panel": {
        **{k: v for k, v in base["panel"].items() if k != "holdout"},
        "holdout": base["panel"]["holdout"]}}, "doses": []}

    doses = [analyze_battery(p) for p in args.doses.split(",")]
    neg = analyze_battery(args.negative) if args.negative else None

    for d in doses:
        c = compare_to_baseline(base, d)
        report["doses"].append(c)
        g = c["gain"]
        print(f"alpha={d['alpha']:+.2f}: rho={c['ordering_rho_raw']:.3f} "
              f"(corrected {c['ordering_rho_attenuation_corrected']:.3f}) | "
              f"beta={g['beta']:.3f} | holdout LL M0={g['m0_holdout']['log_loss']:.4f} "
              f"M1={g['m1_holdout']['log_loss']:.4f} "
              f"M2={c['m2_holdout']['log_loss']:.4f} | "
              f"sharp={c['panel']['sharpness']:.3f} "
              f"cycle={c['panel']['cycle_mass']:.4f} | "
              f"latent resolved={c['latent'].get('frac_resolved')}")

    if neg is not None:
        # match the smallest positive dose by |alpha| for the even/odd split
        pos = min(doses, key=lambda d: abs(abs(d["alpha"]) - abs(neg["alpha"])))
        eo = {}
        for key in ("sharpness", "cycle_mass"):
            eo[key] = even_odd(pos["panel"][key], neg["panel"][key],
                               base["panel"][key])
        rho_pos = float(spearmanr(base["mu"], pos["mu"]).statistic)
        rho_neg = float(spearmanr(base["mu"], neg["mu"]).statistic)
        eo["ordering_rho"] = even_odd(rho_pos, rho_neg, 1.0)
        report["even_odd"] = {"pos_alpha": pos["alpha"], "neg_alpha": neg["alpha"],
                              "components": eo}
        print(f"even/odd (|a|={abs(neg['alpha'])}): " + json.dumps(eo))

    out = args.out or (Path(args.baseline).parent / "compare_report.json")
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()
                    if k not in ("P", "train", "hold", "mu", "sigma2")}
        if isinstance(o, list):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    Path(out).write_text(json.dumps(clean(report), indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
