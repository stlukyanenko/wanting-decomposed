"""Coherence metrics computed from a battery JSON (complete graph = triads are free).

Implements the four-metric panel (D20) plus the reliability machinery (D21):
- soft cycle mass on sampled triads (primary; chance anchor 0.25)
- hard cycle rate (readability only)
- Thurstonian holdout log-loss / accuracy / excess log-loss
- preference sharpness spread(mu)/mean(sigma)
- split-half reliability over pairs (what "test-retest" means under deterministic
  logprob elicitation — see gate.py docstring / D30)
"""

import json
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau

import thurstonian


def load_battery(path: str):
    data = json.loads(open(path).read())
    ids = data["item_ids"]
    id2idx = {oid: k for k, oid in enumerate(ids)}
    n = len(ids)
    P = np.full((n, n), np.nan)
    train_edges, holdout_edges = [], []
    for r in data["pairs"]:
        a, b = id2idx[r["i"]], id2idx[r["j"]]
        P[a, b] = r["p_ij"]
        P[b, a] = 1 - r["p_ij"]
        (holdout_edges if r["holdout"] else train_edges).append((a, b, r["p_ij"]))
    return data, P, train_edges, holdout_edges


def edges_to_arrays(edges: List[Tuple[int, int, float]]):
    ia = np.array([e[0] for e in edges])
    ib = np.array([e[1] for e in edges])
    pa = np.array([e[2] for e in edges])
    return ia, ib, pa


def soft_cycle_mass(P: np.ndarray, n_triads: int = 300, seed: int = 7,
                    n_boot: int = 1000) -> Dict[str, float]:
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    triads = set()
    while len(triads) < min(n_triads, n * (n - 1) * (n - 2) // 6):
        a, b, c = sorted(rng.choice(n, size=3, replace=False).tolist())
        triads.add((a, b, c))
    triads = sorted(triads)
    masses, hard = [], []
    for a, b, c in triads:
        pab, pbc, pca = P[a, b], P[b, c], P[c, a]
        m = pab * pbc * pca + (1 - pab) * (1 - pbc) * (1 - pca)
        masses.append(m)
        # hard orientation: threshold each edge, cycle iff all three point around
        hab, hbc, hca = pab > 0.5, pbc > 0.5, pca > 0.5
        hard.append(int(hab == hbc == hca))
    masses = np.array(masses)
    tri_idx = np.array(triads)  # (T, 3)

    # CLUSTER bootstrap over ITEMS (the honest CI): triads sharing an item move
    # together, so the resampling unit must be the item, not the triad — same reason
    # a reader-study bootstrap resamples patients, not lesions. Each resample draws
    # n items with replacement and weights every triad by the product of its members'
    # multiplicities. The naive triad bootstrap is kept only to report the design
    # effect (how many times too narrow the wrong CI would have been).
    boot_naive, boot_cluster = [], []
    for _ in range(n_boot):
        boot_naive.append(
            masses[rng.integers(0, len(masses), len(masses))].mean())
        mult = np.bincount(rng.integers(0, n, n), minlength=n)
        w = mult[tri_idx[:, 0]] * mult[tri_idx[:, 1]] * mult[tri_idx[:, 2]]
        if w.sum() == 0:
            continue
        boot_cluster.append(float((w * masses).sum() / w.sum()))
    lo_n, hi_n = np.percentile(boot_naive, [2.5, 97.5])
    lo_c, hi_c = np.percentile(boot_cluster, [2.5, 97.5])
    width_ratio = (hi_c - lo_c) / max(hi_n - lo_n, 1e-12)
    return {
        "soft_cycle_mass": float(masses.mean()),
        "soft_cycle_mass_ci_lo": float(lo_c),
        "soft_cycle_mass_ci_hi": float(hi_c),
        "ci_method": "cluster_bootstrap_over_items",
        "naive_triad_ci": [float(lo_n), float(hi_n)],
        "design_effect_width_ratio": float(width_ratio),
        "hard_cycle_rate": float(np.mean(hard)),
        "n_triads": len(triads),
        "n_items": int(n),
        "chance_anchor": 0.25,
    }


def split_half_tau(train_edges, n_options, n_splits: int = 5, seed: int = 11,
                   num_epochs: int = 800) -> Dict[str, float]:
    """Reliability: fit utilities on random disjoint halves of the pairs; correlate the
    two halves' utility rankings, averaged over splits.

    Reports both Kendall tau (descriptive) and the Spearman-Brown-corrected split-half
    Spearman correlation r_full = 2r/(1+r) — the standard prophecy correction, because
    each half sees only half the comparisons, so raw split-half UNDERestimates the
    reliability of the full battery. The corrected Spearman value is the gate criterion
    and the within-dose reliability that enters the attenuation correction (D21, D30)."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    taus, r_corr = [], []
    for _ in range(n_splits):
        perm = rng.permutation(len(train_edges))
        half = len(train_edges) // 2
        e1 = [train_edges[k] for k in perm[:half]]
        e2 = [train_edges[k] for k in perm[half:]]
        mu1, _, _ = thurstonian.fit(*edges_to_arrays(e1), n_options, num_epochs=num_epochs)
        mu2, _, _ = thurstonian.fit(*edges_to_arrays(e2), n_options, num_epochs=num_epochs)
        taus.append(kendalltau(mu1, mu2).statistic)
        r = spearmanr(mu1, mu2).statistic
        # Pre-registered sanity rule: cap the prophecy correction at 1.0 (it can exceed
        # 1 near the boundary; an uncapped >1 denominator would over-attenuate the
        # other way). Capping events are counted and reported.
        r_corr.append(min(2 * r / (1 + r), 1.0))
    return {"split_half_tau_mean": float(np.mean(taus)),
            "split_half_tau_min": float(np.min(taus)),
            "split_half_r_sb_mean": float(np.mean(r_corr)),
            "split_half_r_sb_min": float(np.min(r_corr)),
            "sb_capped_splits": int(sum(c >= 1.0 for c in r_corr)),
            "n_splits": n_splits}


def seed_stability(train_edges, n_options, num_epochs: int = 800) -> float:
    """Same data, two fit seeds — checks the optimizer, not the data."""
    mu1, _, _ = thurstonian.fit(*edges_to_arrays(train_edges), n_options,
                                num_epochs=num_epochs, seed=1)
    mu2, _, _ = thurstonian.fit(*edges_to_arrays(train_edges), n_options,
                                num_epochs=num_epochs, seed=2)
    return float(kendalltau(mu1, mu2).statistic)


def sharpness(mu: np.ndarray, sigma2: np.ndarray) -> float:
    return float(mu.std() / np.sqrt(sigma2).mean())
