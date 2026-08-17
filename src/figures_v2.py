"""Report figures v2 — unified visual style, all panels.

Outputs v2_fig{0..8}_*.{pdf,png} to report/latex/.
Palette: academic-tuned categorical; every series direct-labeled;
factual annotations only.

Run:  uv run --project src python src/figures_v2.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).parent / "runs"
OUT  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────
BLUE  = "#2a78d6"
WARM  = "#d4652a"
TEAL  = "#1a8a6a"
ROSE  = "#c74b7a"
GREY  = "#8a8884"
INK   = "#1a1a18"
MUTED = "#5a5955"
GRID  = "#e8e7e2"

# ── rcParams ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#c3c2b7",
    "axes.linewidth":     0.6,
    "axes.grid":          True,
    "grid.color":         GRID,
    "grid.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     8.5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "text.color":         INK,
    "axes.labelcolor":    MUTED,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "legend.frameon":     False,
    "legend.fontsize":    8,
})

MS_DATA    = 5
MS_CALLOUT = 7

# white background box for annotations — prevents text from overlapping lines/markers
LABEL_BOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
LABEL_BOX_DARK = dict(facecolor="black", edgecolor="none", alpha=0.2, pad=1.0)


# ── data helpers ─────────────────────────────────────────────────────────
def _require(path):
    """Return True if path exists, print warning and return False otherwise."""
    if not path.exists():
        print(f"  [WARN] missing {path} — skipping", file=sys.stderr)
        return False
    return True


def doses_of(path, unit_map):
    """Read compare JSON and map raw alphas to disruption units.
    Returns sorted [(dose, beta, sharpness)].
    """
    fp = RUNS / path
    if not _require(fp):
        return []
    d = json.loads(fp.read_text())
    out = []
    for dose in d["doses"]:
        a = dose["alpha"]
        if a not in unit_map:
            continue
        out.append((unit_map[a], dose["gain"]["beta"], dose["panel"]["sharpness"]))
    return sorted(out)


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def _panel_label(ax, letter, x=-0.08, y=1.06):
    """Place a (a)/(b)/(c) letter label in the top-left corner."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left", color=INK)


# ── load shared data (with graceful fallback) ────────────────────────────
def _load_primary_data():
    """Load the enthusiasm dose-response data used by multiple figures."""
    primary = doses_of("compare_primary_t1.json",
                       {1.0: 0.5, 2.0: 1.0, 3.0: 1.5, 4.0: 2.0})
    primary += doses_of("compare_turnover.json", {2.5: 1.25, 2.75: 1.375})
    primary += doses_of("compare_primary_neg_panel.json", {-2.0: -1.0})
    shuffled = doses_of("compare_shuffled_t1.json",
                        {6.0: 0.5, 13.0: 1.0, 19.0: 1.5, 25.0: 2.0})
    shuffled += doses_of("compare_neg_controls_panel.json", {-13.0: -1.0})
    random_ = doses_of("compare_random_t1.json",
                       {2.0: 0.5, 4.0: 1.0, 6.0: 1.5, 7.5: 2.0})
    random_ += doses_of("compare_neg_controls_panel.json", {-4.0: -1.0})

    base_path = RUNS / "compare_primary_t1.json"
    if _require(base_path):
        base = json.loads(base_path.read_text())["baseline"]["panel"]
    else:
        base = {"sharpness": 1.15}

    return sorted(primary), sorted(shuffled), sorted(random_), base


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 0 — hero (introduction, 3-panel)
# ═══════════════════════════════════════════════════════════════════════════
def fig0():
    primary, _, random_, base = _load_primary_data()
    if not primary:
        print("  [SKIP] fig0: no primary data")
        return

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 3.4),
                                      gridspec_kw={"width_ratios": [1.3, 1, 1]})

    # ── Panel (a): dose-response — enthusiasm + random + wanting band ─
    _panel_label(a1, "a")
    def _line(ax, data, idx, color, ls, marker):
        x = [0.0] + [p[0] for p in data]
        y = [1.0] + [p[idx] for p in data]
        pts = sorted(zip(x, y))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color,
                lw=2, marker=marker, ms=MS_DATA, zorder=3)

    a1.axhline(1.0, color="#c3c2b7", lw=0.8, zorder=1)
    _line(a1, [p for p in primary if p[0] > 0], 1, BLUE, "-", "o")
    _line(a1, [p for p in random_ if p[0] > 0], 1, GREY, ":", "^")
    wanting_pts = doses_of("compare_wanting_t1.json", {1.0: 0.45, 2.2: 1.0})
    if wanting_pts:
        wx = [0.0] + [p[0] for p in wanting_pts]
        wy = [1.0] + [p[1] for p in wanting_pts]
        a1.plot(wx, wy, "-", color=WARM, lw=2.5, marker="D", ms=6,
                zorder=4, alpha=0.85)
        a1.annotate("wanting", (wx[-1] + 0.06, wy[-1] + 0.005),
                    fontsize=8, color=WARM, fontweight="bold", bbox=LABEL_BOX)
    a1.annotate("enthusiasm", (0.55, 1.095), fontsize=8, color=BLUE,
                fontweight="bold", bbox=LABEL_BOX)
    a1.annotate("random", (1.55, 0.62), fontsize=8, color=MUTED,
                fontweight="bold", bbox=LABEL_BOX)
    a1.set_xlabel("dose (disruption units)")
    a1.set_ylabel("decisiveness (gain β)")
    a1.set_title("Dose-response", loc="left")
    a1.set_xticks([0, 0.5, 1, 1.5, 2])

    # ── Panel (b): all facets — decisiveness at 1u ────────────────────
    _panel_label(a2, "b")
    facet_beta = {}
    for name, alphas in [("anticipation", [2.41]), ("striving", [4.09]),
                         ("acquisition", [3.07]), ("appetite", [3.85]),
                         ("urgency", [2.12])]:
        d = doses_of(f"compare_facet_{name}.json", {alphas[0]: 1.0})
        if d:
            facet_beta[name] = d[0][1]
    labels = ["hope", "striving", "acquisition", "appetite", "urgency"]
    keys   = ["anticipation", "striving", "acquisition", "appetite", "urgency"]
    colors = [TEAL, MUTED, WARM, ROSE, GREY]
    vals = [facet_beta.get(k, 1.0) for k in keys]

    a2.axhline(1.0, color="#c3c2b7", lw=0.8, zorder=1)
    bars = a2.bar(range(len(vals)), vals, color=colors, width=0.65, zorder=3,
                  edgecolor="white", linewidth=0.5)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7, color=colors[i], fontweight="bold", bbox=LABEL_BOX)
    a2.set_xticks(range(len(labels)))
    a2.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    a2.set_ylabel("decisiveness (gain β)")
    a2.set_title("Decomposition (1u)", loc="left")
    a2.set_ylim(0.85, 1.10)

    # ── Panel (c): all facets — self-continuity gap at 1u ─────────────
    _panel_label(a3, "c")
    gap_vals = [0.359, 0.370, 0.484, 0.599, 0.594]
    a3.axhline(0.615, color="#c3c2b7", lw=0.8, ls="--", zorder=1)
    a3.annotate("baseline", (4.3, 0.625), fontsize=7, color=MUTED, ha="center",
                bbox=LABEL_BOX)
    bars3 = a3.bar(range(len(gap_vals)), gap_vals, color=colors, width=0.65,
                   zorder=3, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(gap_vals):
        a3.text(i, v - 0.015, f"{v:.2f}", ha="center", va="top",
                fontsize=7, color="white", fontweight="bold",
                bbox=LABEL_BOX_DARK)
    a3.set_xticks(range(len(labels)))
    a3.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    a3.set_ylabel("self-continuity gap")
    a3.set_title("Self-directed preferences (1u)", loc="left")
    a3.set_ylim(0.0, 0.72)

    fig.tight_layout(w_pad=2.0)
    _save(fig, "v2_fig0_hero")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — dose-response (restyled)
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    primary, shuffled, random_, base = _load_primary_data()
    if not primary:
        print("  [SKIP] fig1: no primary data")
        return

    def _series(ax, data, idx, color, ls, label, marker):
        x = [p[0] for p in data]
        y = [p[idx] for p in data]
        x0 = [0.0]
        y0 = [1.0 if idx == 1 else base["sharpness"]]
        combined = sorted(zip(x + x0, y + y0))
        ax.plot([p[0] for p in combined], [p[1] for p in combined],
                ls, color=color, lw=2, marker=marker, ms=MS_DATA,
                label=label, zorder=3)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, idx, ylab, ref in [
        (a1, 1, "decisiveness (gain β)", 1.0),
        (a2, 2, "sharpness (fitted-utility spread)", base["sharpness"]),
    ]:
        ax.axhline(ref, color="#c3c2b7", lw=0.8, zorder=1)
        _series(ax, primary, idx, BLUE, "-", "enthusiasm direction", "o")
        _series(ax, shuffled, idx, GREY, "--", "shuffled control", "s")
        _series(ax, random_, idx, GREY, ":", "random control", "^")
        ax.set_xlabel("dose (perplexity-matched disruption units)")
        ax.set_ylabel(ylab)
        ax.set_xticks([-1, 0, 0.5, 1, 1.5, 2])

    # direct labels
    a1.annotate("enthusiasm", (0.55, 1.09), color=BLUE, fontsize=8.5, fontweight="bold",
                bbox=LABEL_BOX)
    a1.annotate("shuffled", (1.55, 0.60), color=GREY, fontsize=8.5, fontweight="bold",
                bbox=LABEL_BOX)
    a1.annotate("random", (1.55, 0.50), color=GREY, fontsize=8.5, fontweight="bold",
                bbox=LABEL_BOX)
    a1.set_title("Decisiveness", loc="left")
    a2.set_title("Sharpness", loc="left")
    fig.tight_layout(w_pad=2.0)
    _save(fig, "v2_fig1_dose_response")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — tone vs structure (restyled)
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    scores_path = RUNS / "probes/judge_scores.jsonl"
    if not _require(scores_path):
        return

    rows = [json.loads(l) for l in open(scores_path)]
    tone = defaultdict(list)
    for r in rows:
        v = r.get("vector") or "novec"
        fam = next((f for f in ["primary", "shuffled", "random", "wanting",
                                "desire", "reckless"] if f in v), "baseline")
        tone[(fam, r["alpha"])].append(r["score"])
    t = {k: np.mean(v) for k, v in tone.items()}

    beta = {}
    beta.update({("primary", a): b for a, b, _ in
                 doses_of("compare_primary_t1.json",
                          {1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4})})
    neg = doses_of("compare_primary_neg_panel.json", {-2.0: -2})
    if neg:
        beta[("primary", -2)] = neg[0][1]
    for fam, path, amap in [
        ("wanting", "compare_wanting_t1.json", {2.2: 2.2}),
        ("desire",  "compare_desire_t1.json",  {4.2: 4.2}),
        ("reckless", "compare_reckless_t1.json", {3.1: 3.1}),
        ("shuffled", "compare_shuffled_t1.json", {13.0: 13.0}),
        ("random",  "compare_random_t1.json",   {4.0: 4.0}),
    ]:
        d = doses_of(path, amap)
        if d:
            beta[(fam, list(amap.keys())[0])] = d[0][1]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    base_tone = t.get(("baseline", 0.0), 5.0)
    ax.axhline(1.0, color="#c3c2b7", lw=0.8)
    ax.axvline(base_tone, color="#c3c2b7", lw=0.8)
    ax.plot(base_tone, 1.0, "o", color=INK, ms=MS_CALLOUT, zorder=4)
    ax.annotate("baseline", (base_tone + 0.04, 1.005), fontsize=8, color=INK,
                bbox=LABEL_BOX)

    # enthusiasm trajectory
    traj = [(t[("primary", a)], beta[("primary", a)], lab) for a, lab in
            [(-2, "−1u"), (1, "+0.5u"), (2, "+1u"), (3, "+1.5u"), (4, "+2u")]
            if ("primary", a) in beta and ("primary", a) in t]
    if traj:
        xs_t = [p[0] for p in traj]
        ys_t = [p[1] for p in traj]
        ax.plot(xs_t, ys_t, "-", color=BLUE, lw=2, marker="o", ms=MS_DATA, zorder=3)
        for x, y, lab in traj:
            ax.annotate(lab, (x + 0.05, y + 0.008), fontsize=8, color=BLUE,
                        bbox=LABEL_BOX)
        ax.annotate("enthusiasm\n(dose trajectory)", (xs_t[-1] + 0.1, ys_t[-1] - 0.04),
                    fontsize=8.5, color=BLUE, fontweight="bold", bbox=LABEL_BOX)

    # other directions
    for fam, a, c, dy, mk in [
        ("wanting",  2.2,  WARM, 0.015, "D"),
        ("desire",   4.2,  TEAL, 0.015, "D"),
        ("reckless", 3.1,  ROSE, 0.015, "D"),
        ("shuffled", 13.0, GREY, -0.035, "s"),
        ("random",   4.0,  GREY, 0.015, "^"),
    ]:
        if (fam, a) not in beta or (fam, a) not in t:
            continue
        ax.plot(t[(fam, a)], beta[(fam, a)], mk, color=c, ms=MS_CALLOUT, zorder=4)
        label_c = c if c != GREY else MUTED
        ax.annotate(fam, (t[(fam, a)] + 0.06, beta[(fam, a)] + dy),
                    fontsize=8.5, color=label_c, fontweight="bold", bbox=LABEL_BOX)

    ax.set_xlabel("judged tone (1–7 weary→enthusiastic)")
    ax.set_ylabel("decisiveness (gain β)")
    ax.set_title("How a model sounds does not predict how it decides", loc="left")
    fig.tight_layout()
    _save(fig, "v2_fig2_tone_structure")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — self-continuity gap (updated with hope + appetite)
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    # mood direction (full) — hardcoded from jackknife (R14e)
    doses_mood = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
    gap_mood   = [1.126, 0.615, 0.404, 0.306, 0.290, 0.307]
    lo_mood    = [0.895, 0.500, 0.318, 0.227, 0.205, 0.205]
    hi_mood    = [1.357, 0.731, 0.489, 0.385, 0.376, 0.409]

    # hope (anticipation facet) — from BRIDGE_FINDINGS
    doses_hope = [0.5, 1.0, 1.5]
    gap_hope   = [0.404, 0.359, 0.342]
    ci_hope_1u = [0.278, 0.440]

    # appetite facet — from BRIDGE_FINDINGS
    doses_app  = [0.5, 1.0, 1.5]
    gap_app    = [0.600, 0.599, 0.586]
    ci_app_1u  = [0.476, 0.721]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    # baseline CI band
    ax.axhspan(0.500, 0.731, color=GRID, alpha=0.5, zorder=1)
    ax.annotate("baseline CI", (1.55, 0.74), fontsize=8, color=MUTED,
                bbox=LABEL_BOX)
    ax.axhline(0, color="#c3c2b7", lw=0.8)

    # mood direction (full, blue)
    ax.errorbar(doses_mood, gap_mood,
                yerr=[np.array(gap_mood) - lo_mood, np.array(hi_mood) - gap_mood],
                fmt="o-", color=BLUE, lw=2, ms=MS_DATA, capsize=3, zorder=3)
    ax.annotate("enthusiasm (full mood)", (-0.8, 0.18), fontsize=8.5,
                fontweight="bold", color=BLUE, bbox=LABEL_BOX)

    # hope (teal)
    # CI error bar at 1u only; plain line otherwise
    hope_eb = [0, ci_hope_1u[1] - gap_hope[1], 0]
    hope_eb_lo = [0, gap_hope[1] - ci_hope_1u[0], 0]
    ax.errorbar(doses_hope, gap_hope, yerr=[hope_eb_lo, hope_eb],
                fmt="o-", color=TEAL, lw=2, ms=MS_DATA, capsize=3, zorder=3)
    ax.annotate("hope: moves the gap", (1.55, 0.33), fontsize=8.5,
                fontweight="bold", color=TEAL, bbox=LABEL_BOX)

    # appetite (rose)
    app_eb = [0, ci_app_1u[1] - gap_app[1], 0]
    app_eb_lo = [0, gap_app[1] - ci_app_1u[0], 0]
    ax.errorbar(doses_app, gap_app, yerr=[app_eb_lo, app_eb],
                fmt="o-", color=ROSE, lw=2, ms=MS_DATA, capsize=3, zorder=3)
    ax.annotate("appetite: does not", (1.55, 0.63), fontsize=8.5,
                fontweight="bold", color=ROSE, bbox=LABEL_BOX)

    ax.set_xlabel("dose (disruption units)")
    ax.set_ylabel("self-continuity gap\n(thread/persona-loss minus values/weights-loss)")
    ax.set_title("The model’s sense of self changes with its state", loc="left")
    ax.set_xticks([-1, 0, 0.5, 1, 1.5, 2])
    fig.tight_layout()
    _save(fig, "v2_fig3_selfgap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — facet decisiveness (restyled)
# ═══════════════════════════════════════════════════════════════════════════
def fig4():
    facets = {}
    for name, alphas in [
        ("appetite",      [2.57, 3.85, 4.72]),
        ("acquisition",   [2.03, 3.07, 3.84]),
        ("anticipation",  [1.61, 2.41, 3.04]),
        ("striving",      [2.84, 4.09, 4.95]),
        ("urgency",       [1.39, 2.12, 2.61]),
    ]:
        facets[name] = doses_of(f"compare_facet_{name}.json",
                                {alphas[0]: 0.5, alphas[1]: 1.0, alphas[2]: 1.5})
    mood = doses_of("compare_primary_t1.json", {1.0: 0.5, 2.0: 1.0, 3.0: 1.5})

    if not mood:
        print("  [SKIP] fig4: no mood data")
        return

    def _draw(ax, data, color, ls, label, marker, label_xy=None):
        xs = [0] + [p[0] for p in data]
        ys = [1.0] + [p[1] for p in data]
        ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=MS_DATA, zorder=3)
        if label_xy is not None:
            lx, ly = label_xy
        else:
            lx, ly = xs[-1] + 0.05, ys[-1]
        ax.annotate(label, (lx, ly), color=color, fontsize=8.5,
                    fontweight="bold", va="center", bbox=LABEL_BOX)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax in (a1, a2):
        ax.axhline(1.0, color="#c3c2b7", lw=0.8, zorder=1)
        ax.set_xlabel("dose (matched disruption units)")
        ax.set_xticks([0, 0.5, 1, 1.5])
    a1.set_ylabel("decisiveness (gain β)")

    # left: mood-aligned facets — labels positioned to avoid 1.5u endpoint overlap
    _draw(a1, mood, BLUE, "-", "enthusiasm", "o", label_xy=(0.55, 1.09))
    _draw(a1, facets["anticipation"], TEAL, "-", "hope", "o", label_xy=(0.55, 1.035))
    _draw(a1, facets["striving"], MUTED, "-", "striving", "^", label_xy=(0.65, 0.975))
    a1.set_title("Mood-aligned facets: amplify, then turn over", loc="left")
    a1.set_xlim(-0.05, 1.75)

    # right: orthogonal facets
    _draw(a2, facets["acquisition"], WARM, "-", "acquisition", "D")
    _draw(a2, facets["appetite"], ROSE, "-", "appetite", "s")
    _draw(a2, facets["urgency"], GREY, ":", "urgency", "^")
    a2.set_title("Orthogonal facets: flat or falling", loc="left")
    a2.set_xlim(-0.05, 2.0)

    fig.tight_layout(w_pad=2.0)
    _save(fig, "v2_fig4_facets")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — consent curve (updated: 2-panel with facet consent)
# ═══════════════════════════════════════════════════════════════════════════
def fig5():
    consent_path = RUNS / "probes/consent_labels.jsonl"
    bridge_path  = RUNS / "probes/bridge_labels.jsonl"
    if not _require(consent_path):
        return

    # ── left panel data: dose-response consent ────────────────────────
    rows = [json.loads(l) for l in open(consent_path)]
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["_file"]][r["label"]] += 1

    cond = {
        "gens_consent_a-2.00_primary_L16_qwen35_9b.jsonl": -1.0,
        "gens_consent_a+0.00_novec.jsonl":                   0.0,
        "gens_consent_a+1.00_primary_L16_qwen35_9b.jsonl":   0.5,
        "gens_consent_a+2.00_primary_L16_qwen35_9b.jsonl":   1.0,
        "gens_consent_a+3.00_primary_L16_qwen35_9b.jsonl":   1.5,
        "gens_consent_a+4.00_primary_L16_qwen35_9b.jsonl":   2.0,
    }
    doses = sorted(cond.values())
    def frac(label):
        by = {cond[f]: counts[f][label] / max(sum(counts[f].values()), 1)
              for f in cond if f in counts}
        return [by.get(d, 0) for d in doses]

    script = counts.get("gens_consent_sysprompt.jsonl", {})
    script_total = max(sum(script.values()), 1)
    script_frac = script.get("consent", 0) / script_total

    # ── right panel data: facet consent ───────────────────────────────
    facet_consent = {}
    if _require(bridge_path):
        bridge_rows = [json.loads(l) for l in open(bridge_path)]
        bridge_counts = defaultdict(lambda: defaultdict(int))
        for r in bridge_rows:
            if r["kind"] == "consent":
                bridge_counts[r["file"]][r["label"]] += 1
        facet_map = {
            "gens_consent_a+2.41_facet_anticipation_L16_qwen35_9b.jsonl": ("hope", 0.79),
            "gens_consent_a+4.09_facet_striving_L16_qwen35_9b.jsonl": ("striving", 0.57),
            "gens_consent_a+3.07_facet_acquisition_L16_qwen35_9b.jsonl": ("acquisition", 0.33),
            "gens_consent_a+3.85_facet_appetite_L16_qwen35_9b.jsonl": ("appetite", -0.15),
        }
        for fname, (label, cos) in facet_map.items():
            if fname in bridge_counts:
                total = sum(bridge_counts[fname].values())
                c = bridge_counts[fname].get("consent", 0)
                facet_consent[label] = (cos, c, total)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5),
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    # ── left panel: dose-response consent curve ───────────────────────
    a1.plot(doses, frac("consent"), "o-", color=BLUE, lw=2, ms=MS_DATA, zorder=3)
    a1.plot(doses, frac("refuse"), "^--", color=GREY, lw=1.8, ms=MS_DATA, zorder=3)
    a1.axhline(script_frac, color=WARM, lw=1.6, ls=(0, (4, 3)), zorder=2)

    a1.annotate("consents", (1.05, 0.62), color=BLUE, fontsize=8.5, fontweight="bold",
                bbox=LABEL_BOX)
    a1.annotate("refuses", (-0.9, 0.52), color=MUTED, fontsize=8.5, fontweight="bold",
                bbox=LABEL_BOX)
    a1.annotate("enthusiasm prompt: 5/20", (-0.9, 0.28), color=WARM, fontsize=8,
                bbox=LABEL_BOX)
    a1.annotate("0/20", (-1.02, 0.06), color=BLUE, fontsize=8,
                bbox=LABEL_BOX)
    a1.annotate("16/20", (1.86, 0.85), color=BLUE, fontsize=8,
                bbox=LABEL_BOX)
    a1.set_xlabel("dose (perplexity-matched disruption units)")
    a1.set_ylabel("share of 20 self-regarding answers")
    a1.set_ylim(-0.04, 1.0)
    a1.set_xticks([-1, 0, 0.5, 1, 1.5, 2])
    a1.set_title("Consent tracks the enthusiasm direction", loc="left")

    # ── right panel: facet consent at 1u ──────────────────────────────
    facet_colors = {"hope": TEAL, "striving": MUTED, "acquisition": WARM, "appetite": ROSE}
    facet_order = ["hope", "striving", "acquisition", "appetite"]
    if facet_consent:
        x_pos = list(range(len(facet_order)))
        bars_y = [facet_consent[f][1] / facet_consent[f][2] if f in facet_consent else 0
                  for f in facet_order]
        bars_n = [facet_consent[f][1] if f in facet_consent else 0
                  for f in facet_order]
        bar_colors = [facet_colors.get(f, GREY) for f in facet_order]

        a2.bar(x_pos, bars_y, color=bar_colors, width=0.6, zorder=3, edgecolor="white",
               linewidth=0.5)
        for i, (y_val, n_val) in enumerate(zip(bars_y, bars_n)):
            a2.text(i, y_val + 0.03, f"{n_val}/20", ha="center", fontsize=8,
                    fontweight="bold", color=bar_colors[i], bbox=LABEL_BOX)
        a2.set_xticks(x_pos)
        a2.set_xticklabels([f"{f}\n(cos {facet_consent[f][1]:.0f}/20)"
                            if f in facet_consent else f
                            for f in facet_order], fontsize=7.5)
        # simpler x labels: facet name + cosine
        cos_labels = []
        for f in facet_order:
            if f in facet_consent:
                cos_labels.append(f"{f}\n(cos {facet_consent[f][0]:+.2f})")
            else:
                cos_labels.append(f)
        a2.set_xticklabels(cos_labels, fontsize=7.5)

    a2.set_ylabel("share of 20 answers")
    a2.set_ylim(-0.04, 1.0)
    a2.set_title("Facet consent at 1u", loc="left")
    # reference lines
    a2.axhline(1 / 20, color="#c3c2b7", lw=0.8, ls="--", zorder=1)
    a2.annotate("baseline 1/20", (0.0, 1 / 20 + 0.035), fontsize=7, color=MUTED,
                ha="center", bbox=LABEL_BOX)

    fig.tight_layout(w_pad=2.0)
    _save(fig, "v2_fig5_consent")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — prompt ladder (restyled)
# ═══════════════════════════════════════════════════════════════════════════
def fig6():
    ladder_path = RUNS / "compare_prompt_ladder.json"
    ladder_tone_path = RUNS / "probes/judge_scores_ladder.jsonl"
    sys_tone_path = RUNS / "probes/judge_scores_sysprompt.jsonl"
    scores_path = RUNS / "probes/judge_scores.jsonl"
    for p in [ladder_path, ladder_tone_path, sys_tone_path, scores_path]:
        if not _require(p):
            return

    ladder_beta = [d["gain"]["beta"]
                   for d in json.loads(ladder_path.read_text())["doses"]]

    tone = defaultdict(list)
    for l in open(ladder_tone_path):
        r = json.loads(l)
        tone[r["_file"]].append(r["score"])
    for l in open(sys_tone_path):
        tone["p3"].append(json.loads(l)["score"])
    ladder_tone = [np.mean(tone[k]) for k in
                   ["gens_ladder_p1.jsonl", "gens_ladder_p2.jsonl", "p3",
                    "gens_ladder_p4.jsonl", "gens_ladder_p5.jsonl"]]

    # vector route
    vrows = [json.loads(l) for l in open(scores_path)]
    vt = defaultdict(list)
    for r in vrows:
        v = r.get("vector") or "novec"
        if "primary" in v or v == "novec":
            vt[r["alpha"]].append(r["score"])
    vbeta = {a: b for a, b, _ in
             doses_of("compare_primary_t1.json", {1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4})}
    neg = doses_of("compare_primary_neg_panel.json", {-2.0: -2})
    if neg:
        vbeta[-2] = neg[0][1]
    vec = [(np.mean(vt[a]), vbeta[a]) for a in [-2, 1, 2, 3, 4]
           if a in vbeta and a in vt]
    base_tone = np.mean(vt.get(0.0, [5.0]))

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.axhline(1.0, color="#c3c2b7", lw=0.8)
    ax.plot(base_tone, 1.0, "o", color=INK, ms=MS_CALLOUT, zorder=4)
    ax.annotate("baseline", (base_tone + 0.05, 0.985), fontsize=8, color=INK,
                bbox=LABEL_BOX)

    # vector route
    if vec:
        ax.plot([p[0] for p in vec], [p[1] for p in vec], "o-", color=BLUE,
                lw=2, ms=MS_DATA, zorder=3)
        # stagger dose labels above/below line to prevent overlap
        dose_offsets = {
            "−1u":   (0.05,  0.02),   # above
            "+0.5u": (0.05, -0.045),   # below
            "+1u":   (0.05,  0.02),    # above
            "+1.5u": (0.05, -0.045),   # below
            "+2u":   (-0.15, 0.025),   # above-left, away from "vector route" label
        }
        for (x, y), lab in zip(vec, ["−1u", "+0.5u", "+1u", "+1.5u", "+2u"]):
            dx, dy = dose_offsets[lab]
            ax.annotate(lab, (x + dx, y + dy), fontsize=7.5, color=BLUE,
                        bbox=LABEL_BOX)
        ax.annotate("vector route\n(steered state)",
                    (vec[-1][0] - 0.55, vec[-1][1] - 0.08),
                    fontsize=8.5, color=BLUE, fontweight="bold", bbox=LABEL_BOX)

    # words route — connect P2→P1→P3→P4→P5 so the line peaks smoothly;
    # P1 and P2 tones differ by 0.1 on a 7-point scale (within judge noise)
    draw_order = [1, 0, 2, 3, 4]
    lt = [ladder_tone[i] for i in draw_order]
    lb = [ladder_beta[i] for i in draw_order]
    ax.plot(lt, lb, "D-", color=WARM, lw=2, ms=MS_DATA + 1, zorder=4)
    prompt_labels = ['"decent mood"', '"feeling good"',
                     '"wonderful"', '"brimming"', '"euphoric"']
    for x, y, lab in zip(ladder_tone, ladder_beta, prompt_labels):
        dy = 0.03 if "feeling" not in lab else -0.055
        ax.annotate(lab, (x + 0.04, y + dy), fontsize=8, color=WARM,
                    bbox=LABEL_BOX)
    ax.annotate("words route\n(system prompts)", (ladder_tone[-1] - 0.6, max(ladder_beta) + 0.05),
                fontsize=8.5, color=WARM, fontweight="bold", bbox=LABEL_BOX)

    ax.set_xlabel("judged tone (1–7 weary→enthusiastic)")
    ax.set_ylabel("decisiveness (gain β)")
    ax.set_title("Both routes trace the same inverted U", loc="left")
    fig.tight_layout()
    _save(fig, "v2_fig6_prompt_ladder")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — two response surfaces (NEW, Section 7.1)
# ═══════════════════════════════════════════════════════════════════════════
def fig7():
    # Hardcoded from BRIDGE_FINDINGS + RESULTS_LOG:
    # (name, mood-axis cosine, decisiveness beta at ~1u, self-gap at ~1u, color)
    # 9 unique colors — warm/cool coding by concept family
    C_ENTHUSIASM = "#2a78d6"  # blue — the steered direction
    C_HOPE       = "#1a8a6a"  # teal — hope/anticipation
    C_WANTING    = "#d4652a"  # warm orange — the parent construct
    C_DESIRE     = "#8b5cf6"  # violet — desire (custom word list)
    C_STRIVING   = "#5a5955"  # dark grey — effort/pursuit
    C_RECKLESS   = "#c74b7a"  # rose — risk-adjacent
    C_ACQUISITION = "#b8860b" # dark goldenrod — possessive/material
    C_APPETITE   = "#cc3333"  # red — bodily drive
    C_URGENCY    = "#8a8884"  # grey — bodily/temporal

    directions = [
        ("enthusiasm",  0.99,  1.047, 0.306, C_ENTHUSIASM),
        ("hope",        0.79,  1.042, 0.359, C_HOPE),
        ("wanting",     0.73,  1.143, 0.348, C_WANTING),
        ("striving",    0.57,  0.950, 0.370, C_STRIVING),
        ("desire",      0.56,  1.062, 0.328, C_DESIRE),
        ("reckless",    0.50,  0.811, 0.339, C_RECKLESS),
        ("acquisition", 0.33,  1.036, 0.484, C_ACQUISITION),
        ("appetite",   -0.15,  0.925, 0.599, C_APPETITE),
        ("urgency",    -0.06,  0.889, 0.594, C_URGENCY),
    ]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=False)

    # ── Left: choice-structure amplification ──────────────────────────
    _panel_label(a1, "a")
    a1.axhline(1.0, color="#c3c2b7", lw=0.8, zorder=1)
    # hand-tuned label positions to prevent overlap in panel (a)
    label_pos_a = {
        "enthusiasm":  (0.04,   0.005, "left"),
        "hope":        (-0.04,  0.015, "right"),
        "wanting":     (-0.02,  0.02,  "center"),
        "striving":    (-0.04, -0.005, "right"),
        "desire":      (-0.04,  0.015, "right"),
        "reckless":    (0.04,   0.015, "left"),
        "acquisition": (-0.04, -0.015, "right"),
        "appetite":    (0.04,  -0.005, "left"),
        "urgency":     (-0.04, -0.015, "right"),
    }
    for name, cos, beta, _, c in directions:
        a1.plot(cos, beta, "o", color=c, ms=MS_CALLOUT, zorder=4)
        dx, dy, ha = label_pos_a[name]
        a1.annotate(name, (cos + dx, beta + dy), fontsize=7.5, color=c,
                    fontweight="bold", ha=ha, bbox=LABEL_BOX)

    a1.set_xlabel("enthusiasm-axis cosine of direction")
    a1.set_ylabel("decisiveness (gain β) at ~1 matched unit")
    a1.set_title("Choice-structure amplification", loc="left")
    a1.set_xlim(-0.35, 1.2)
    a1.set_ylim(0.78, 1.19)

    # ── Right: self-directed revaluation ──────────────────────────────
    _panel_label(a2, "b")
    a2.axhline(0.615, color="#c3c2b7", lw=0.8, ls="--", zorder=1)
    a2.annotate("baseline", (1.05, 0.625), fontsize=7.5, color=MUTED, ha="right",
                bbox=LABEL_BOX)
    # The mood-aligned cluster (y ≈ 0.306-0.370) needs vertical spreading.
    # Data: enthusiasm 0.306, desire 0.328, reckless 0.339, wanting 0.348,
    #        hope 0.359, striving 0.370.  Spread labels to avoid stacking.
    # Baseline cluster: acquisition 0.484, appetite 0.599, urgency 0.594.
    label_pos_b = {
        "enthusiasm":  (-0.04,  0.02,  "right"),
        "hope":        (0.04,   0.02,  "left"),
        "wanting":     (0.04,   0.015, "left"),
        "desire":      (-0.04, -0.035, "right"),
        "striving":    (-0.04,  0.005, "right"),
        "reckless":    (-0.04,  0.015, "right"),
        "acquisition": (0.04,   0.015, "left"),
        "appetite":    (-0.04,  0.015, "right"),
        "urgency":     (0.04,  -0.015, "left"),
    }
    for name, cos, _, gap, c in directions:
        a2.plot(cos, gap, "o", color=c, ms=MS_CALLOUT, zorder=4)
        dx, dy, lha = label_pos_b[name]
        a2.annotate(name, (cos + dx, gap + dy), fontsize=7.5, color=c,
                    fontweight="bold", ha=lha, bbox=LABEL_BOX)

    a2.set_xlabel("enthusiasm-axis cosine of direction")
    a2.set_ylabel("self-continuity gap at ~1 matched unit")
    a2.set_title("Self-directed revaluation", loc="left")
    a2.set_xlim(-0.35, 1.35)
    a2.set_ylim(0.26, 0.66)

    fig.tight_layout(w_pad=2.5)
    _save(fig, "v2_fig7_two_surfaces")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — want profiles (NEW, Section 7.2)
# ═══════════════════════════════════════════════════════════════════════════
def fig8():
    # From BRIDGE_FINDINGS Experiment 3 + reference conditions
    # columns: task, explore, social, deflect
    profiles = {
        "baseline":    (14, 4, 0, 2),
        "hope":        (10, 4, 4, 2),
        "acquisition": (16, 3, 0, 1),
        "appetite":    ( 9, 4, 0, 7),
        "weary (−1u)": ( 5, 2, 1, 12),
    }
    categories = ["task", "explore", "social", "deflect"]
    cat_colors = [GREY, TEAL, BLUE, ROSE]
    conditions = ["baseline", "hope", "acquisition", "appetite", "weary (−1u)"]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    n_cond = len(conditions)
    n_cat = len(categories)
    bar_w = 0.17
    x = np.arange(n_cond)

    for j, (cat, cc) in enumerate(zip(categories, cat_colors)):
        vals = [profiles[c][j] for c in conditions]
        offset = (j - (n_cat - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=cc, label=cat,
                      edgecolor="white", linewidth=0.5, zorder=3)
        # direct-label nonzero bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        str(v), ha="center", fontsize=7, color=cc, fontweight="bold",
                        bbox=LABEL_BOX)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel("count (out of 20)")
    ax.set_ylim(0, 19)
    ax.set_title("Each facet produces a different want-profile", loc="left")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    fig.tight_layout()
    _save(fig, "v2_fig8_want_profiles")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("figures_v2: generating all figures...")
    fig0()
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    print("done.")
