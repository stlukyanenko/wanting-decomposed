"""Thurstonian utility fitter.

Ported with light adaptation from the CAIS Utility Engineering codebase
(resources/emergent-values/utility_analysis/compute_utilities/utility_models/thurstonian/utils.py,
Mazeika et al. 2025, MIT-licensed) — decision D6/D14 in ideas/DECISIONS_LOG.md.
Changes from upstream: consumes plain arrays instead of their PreferenceGraph, seedable
init, silent training loop, predict() helper. The per-epoch z-scoring of mu is kept
verbatim — it is why all downstream measures must be differential (D22).
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def fit(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    prob_a: np.ndarray,
    n_options: int,
    num_epochs: int = 1000,
    learning_rate: float = 0.01,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit mu/sigma^2 per option from soft pairwise labels P(a > b).

    Returns (mu, sigma2, train_metrics). mu is z-scored by construction.
    """
    g = torch.Generator().manual_seed(seed)
    mu = torch.nn.Parameter(torch.randn(n_options, generator=g) * 0.01)
    s = torch.nn.Parameter(torch.randn(n_options, generator=g) * 0.01)
    optimizer = torch.optim.Adam([mu, s], lr=learning_rate)

    ia = torch.tensor(idx_a, dtype=torch.long)
    ib = torch.tensor(idx_b, dtype=torch.long)
    labels = torch.tensor(prob_a, dtype=torch.float32)
    normal = torch.distributions.Normal(0, 1)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        mu_n = (mu - mu.mean()) / (mu.std() + 1e-5)
        scale = 1 / (mu.std() + 1e-5)
        sigma2_n = torch.exp(s) * scale**2
        z = (mu_n[ia] - mu_n[ib]) / torch.sqrt(sigma2_n[ia] + sigma2_n[ib] + 1e-5)
        p = normal.cdf(z).clamp(1e-5, 1 - 1e-5)
        loss = F.binary_cross_entropy(p, labels, reduction="mean")
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mu_n = (mu - mu.mean()) / (mu.std() + 1e-5)
        scale = 1 / (mu.std() + 1e-5)
        sigma2_n = torch.exp(s) * scale**2
    mu_np = mu_n.numpy()
    sigma2_np = sigma2_n.numpy()
    metrics = evaluate(idx_a, idx_b, prob_a, mu_np, sigma2_np)
    return mu_np, sigma2_np, metrics


def predict(idx_a, idx_b, mu, sigma2) -> np.ndarray:
    """Model P(a > b) for index arrays under fitted utilities."""
    from scipy.stats import norm

    z = (mu[idx_a] - mu[idx_b]) / np.sqrt(sigma2[idx_a] + sigma2[idx_b] + 1e-5)
    return norm.cdf(z)


def evaluate(idx_a, idx_b, prob_a, mu, sigma2) -> Dict[str, float]:
    """Log-loss / accuracy / excess log-loss of fitted utilities on given edges.

    Excess log loss = log loss minus the entropy of the observed soft labels — the
    red team's noise-robust fit metric (D21): raw log loss rises under pure label
    noise even when the model is perfect; the excess stays flat.
    """
    p = np.clip(predict(idx_a, idx_b, mu, sigma2), 1e-5, 1 - 1e-5)
    y = np.asarray(prob_a, dtype=np.float64)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    yc = np.clip(y, 1e-5, 1 - 1e-5)
    label_entropy = float(-np.mean(yc * np.log(yc) + (1 - yc) * np.log(1 - yc)))
    accuracy = float(np.mean((p >= 0.5) == (y >= 0.5)))
    return {
        "log_loss": log_loss,
        "accuracy": accuracy,
        "label_entropy": label_entropy,
        "excess_log_loss": log_loss - label_entropy,
    }
