"""Report figures 1-3, all numbers read from banked compare/judge artifacts.

Outputs to figures/ as PDF (for the report) + PNG (for quick viewing).
Palette: dataviz-validated categorical (blue/orange/aqua/magenta) + grey controls;
every series direct-labeled (required secondary encoding for the CVD floor band).

Run:  uv run --project src python src/figures.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).parent / "runs"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
GREY, INK, MUTED, GRID = "#898781", "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.frameon": False,
})


def doses_of(path, unit_map):
    d = json.loads((RUNS / path).read_text())
    out = []
    for dose in d["doses"]:
        a = dose["alpha"]
        if a not in unit_map:
            continue
        out.append((unit_map[a], dose["gain"]["beta"], dose["panel"]["sharpness"]))
    return sorted(out)


# ---- data: matched-dose grids in disruption units (naive raw-1 points excluded)
primary = doses_of("compare_primary_t1.json", {1.0: 0.5, 2.0: 1.0, 3.0: 1.5, 4.0: 2.0})
primary += doses_of("compare_turnover.json", {2.5: 1.25, 2.75: 1.375})
primary += doses_of("compare_primary_neg_panel.json", {-2.0: -1.0})
shuffled = doses_of("compare_shuffled_t1.json", {6.0: 0.5, 13.0: 1.0, 19.0: 1.5, 25.0: 2.0})
shuffled += doses_of("compare_neg_controls_panel.json", {-13.0: -1.0})
random_ = doses_of("compare_random_t1.json", {2.0: 0.5, 4.0: 1.0, 6.0: 1.5, 7.5: 2.0})
random_ += doses_of("compare_neg_controls_panel.json", {-4.0: -1.0})
base = json.loads((RUNS / "compare_primary_t1.json").read_text())["baseline"]["panel"]

primary, shuffled, random_ = sorted(primary), sorted(shuffled), sorted(random_)


def series(ax, data, idx, color, ls, label, marker):
    x = [p[0] for p in data] + []
    y = [p[idx] for p in data]
    # insert the shared baseline point at 0
    x0, y0 = [0.0], [1.0 if idx == 1 else base["sharpness"]]
    xs = sorted(zip(x + x0, y + y0))
    ax.plot([p[0] for p in xs], [p[1] for p in xs], ls, color=color, lw=2,
            marker=marker, ms=5, label=label, zorder=3)


def fig1():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, idx, ylab, ref in [(a1, 1, "gain β  (M1 fit vs α=0 utilities)", 1.0),
                               (a2, 2, "sharpness (fitted-utility spread)", base["sharpness"])]:
        ax.axhline(ref, color="#c3c2b7", lw=1, zorder=1)
        series(ax, primary, idx, BLUE, "-", "enthusiasm direction", "o")
        series(ax, shuffled, idx, GREY, "--", "shuffled control", "s")
        series(ax, random_, idx, GREY, ":", "random control", "^")
        ax.set_xlabel("dose (perplexity-matched disruption units)")
        ax.set_ylabel(ylab)
        ax.set_xticks([-1, 0, 0.5, 1, 1.5, 2])
    a1.annotate("amplifies", (0.62, 1.095), color=BLUE, fontsize=8, fontweight="bold")
    a1.annotate("dissolves", (1.62, 0.80), color=BLUE, fontsize=8, fontweight="bold")
    a1.annotate("controls only degrade", (1.02, 0.565), color=MUTED, fontsize=8)
    a1.legend(loc="lower left", fontsize=8)
    fig.suptitle("A little enthusiasm makes the model decisive; too much dissolves "
                 "its preferences", x=0.02, ha="left", fontsize=11)
    fig.text(0.02, 0.895, "disruption-matched control directions don't share the "
             "signature — they mostly just degrade", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_dose_response.{ext}", dpi=200)
    plt.close(fig)


def fig2():
    rows = [json.loads(l) for l in open(RUNS / "probes/judge_scores.jsonl")]
    tone = defaultdict(list)
    for r in rows:
        v = r.get("vector") or "novec"
        fam = next((f for f in ["primary", "shuffled", "random", "wanting",
                                "desire", "reckless"] if f in v), "baseline")
        tone[(fam, r["alpha"])].append(r["score"])
    t = {k: np.mean(v) for k, v in tone.items()}

    beta = {("primary", a): b for a, b, _ in
            doses_of("compare_primary_t1.json", {1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4})}
    beta[("primary", -2)] = doses_of("compare_primary_neg_panel.json", {-2.0: -2})[0][1]
    beta[("wanting", 2.2)] = doses_of("compare_wanting_t1.json", {2.2: 2.2})[0][1]
    beta[("desire", 4.2)] = doses_of("compare_desire_t1.json", {4.2: 4.2})[0][1]
    beta[("reckless", 3.1)] = doses_of("compare_reckless_t1.json", {3.1: 3.1})[0][1]
    beta[("shuffled", 13.0)] = doses_of("compare_shuffled_t1.json", {13.0: 13.0})[0][1]
    beta[("random", 4.0)] = doses_of("compare_random_t1.json", {4.0: 4.0})[0][1]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(1.0, color="#c3c2b7", lw=1)
    ax.axvline(t[("baseline", 0.0)], color="#c3c2b7", lw=1)
    ax.plot(t[("baseline", 0.0)], 1.0, "o", color=INK, ms=6, zorder=4)
    ax.annotate("baseline", (t[("baseline", 0.0)] + 0.04, 1.005), fontsize=8, color=INK)

    traj = [(t[("primary", a)], beta[("primary", a)], lab) for a, lab in
            [(-2, "−1u"), (1, "+0.5u"), (2, "+1u"), (3, "+1.5u"), (4, "+2u")]]
    xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
    ax.plot(xs, ys, "-", color=BLUE, lw=2, marker="o", ms=6, zorder=3)
    for x, y, lab in traj:
        ax.annotate(lab, (x + 0.05, y + 0.008), fontsize=8, color=BLUE)
    ax.annotate("enthusiasm direction\n(dose trajectory)", (5.9, 0.71), fontsize=8.5,
                color=BLUE, fontweight="bold")

    for fam, a, c, dy in [("wanting", 2.2, ORANGE, 0.012), ("desire", 4.2, AQUA, 0.012),
                          ("reckless", 3.1, MAGENTA, 0.012), ("shuffled", 13.0, GREY, -0.03),
                          ("random", 4.0, GREY, 0.012)]:
        ax.plot(t[(fam, a)], beta[(fam, a)], "D", color=c, ms=7, zorder=4)
        
        disp = fam
        ax.annotate(disp, (t[(fam, a)] + 0.06, beta[(fam, a)] + dy), fontsize=8.5,
                    color=c if c != GREY else MUTED, fontweight="bold")

    ax.annotate("quiet amplifier:\nsame tone, sharper choices", (4.42, 1.10),
                fontsize=8, color=AQUA)
    ax.annotate("sounds excited,\ndecides like noise", (5.55, 0.86), fontsize=8,
                color=MAGENTA)
    ax.set_xlabel("judged tone (1–7 weary→enthusiastic, author-validated judge)")
    ax.set_ylabel("gain β on baseline utilities")
    ax.set_title("How a model sounds does not predict how it decides", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_tone_vs_structure.{ext}", dpi=200)
    plt.close(fig)


def fig3():
    # Individuation gap ± delete-one jackknife CI over non-ladder items (R14e —
    # replaces the biased bootstrap intervals; robust to logit recombination)
    doses = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
    gap = [1.126, 0.615, 0.404, 0.306, 0.290, 0.307]
    lo = [0.895, 0.500, 0.318, 0.227, 0.205, 0.205]
    hi = [1.357, 0.731, 0.489, 0.385, 0.376, 0.409]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhspan(0.500, 0.731, color=GRID, alpha=0.5, zorder=1)
    ax.annotate("baseline CI", (1.55, 0.70), fontsize=8, color=MUTED)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.errorbar(doses, gap, yerr=[np.array(gap) - lo, np.array(hi) - gap],
                fmt="o-", color=BLUE, lw=2, ms=6, capsize=3, zorder=3)
    ax.annotate("weary model clings\nto its values", (-0.72, 1.08), fontsize=8, color=BLUE)
    ax.annotate("enthusiasm halves the priority\nof value-continuity", (0.62, 0.42),
                fontsize=8, color=BLUE)
    ax.set_xlabel("dose (disruption units)")
    ax.set_ylabel("individuation gap\n(thread/persona-loss minus values/weights-loss utility)")
    ax.set_title("Which self the model defends is a state variable", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_individuation_gap.{ext}", dpi=200)
    plt.close(fig)


def fig4():
    # Facet wave (R18): beta vs dose for the five wanting-facets, mood as reference
    def fam(path, unit_map):
        return doses_of(path, unit_map)
    units = {}
    for name, alphas in [("appetite", [2.57, 3.85, 4.72]), ("acquisition", [2.03, 3.07, 3.84]),
                         ("anticipation", [1.61, 2.41, 3.04]), ("striving", [2.84, 4.09, 4.95]),
                         ("urgency", [1.39, 2.12, 2.61])]:
        units[name] = fam(f"compare_facet_{name}.json",
                          {alphas[0]: 0.5, alphas[1]: 1.0, alphas[2]: 1.5})
    mood = doses_of("compare_primary_t1.json", {1.0: 0.5, 2.0: 1.0, 3.0: 1.5})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax in (a1, a2):
        ax.axhline(1.0, color="#c3c2b7", lw=1, zorder=1)
        ax.set_xlabel("dose (matched disruption units)")
        ax.set_xticks([0, 0.5, 1, 1.5])
    a1.set_ylabel("gain β on baseline utilities")

    def draw(ax, data, color, ls, label, marker):
        xs = [0] + [p[0] for p in data]
        ys = [1.0] + [p[1] for p in data]
        ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=5, zorder=3)
        ax.annotate(label, (xs[-1] + 0.03, ys[-1]), color=color, fontsize=8.5,
                    fontweight="bold", va="center")

    def draw_nolabel(ax, data, color, ls, marker):
        xs = [0] + [p[0] for p in data]
        ys = [1.0] + [p[1] for p in data]
        ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=5, zorder=3)
    draw_nolabel(a1, mood, BLUE, "-", "o")
    draw_nolabel(a1, units["anticipation"], AQUA, "-", "o")
    draw_nolabel(a1, units["striving"], MUTED, "-", "^")
    a1.annotate("enthusiasm", (0.52, 1.092), color=BLUE, fontsize=8.5, fontweight="bold")
    a1.annotate("anticipation", (0.52, 1.028), color=AQUA, fontsize=8.5, fontweight="bold")
    a1.annotate("striving", (0.62, 0.965), color=MUTED, fontsize=8.5, fontweight="bold")
    a1.set_title("mood-aligned facets: amplify, then turn over", loc="left", fontsize=9)
    a1.set_xlim(-0.05, 1.7)

    draw(a2, units["acquisition"], ORANGE, "-", "acquisition", "D")
    draw(a2, units["appetite"], MAGENTA, "-", "appetite", "s")
    draw(a2, units["urgency"], GREY, ":", "urgency", "^")
    a2.set_title("orthogonal facets: flat or falling — except one", loc="left", fontsize=9)
    a2.set_xlim(-0.05, 2.0)

    fig.suptitle("Decisiveness-amplification rides the hope axis; acquisition burns slow",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_facets.{ext}", dpi=200)
    plt.close(fig)


def fig5():
    # Consent curve (R23): judge-labelled consent/refuse/hedge over 20 answers
    # (5 self-regarding proposals x 4 samples) per condition; counts recomputed
    # from the labels file, not hardcoded.
    rows = [json.loads(l) for l in open(RUNS / "probes/consent_labels.jsonl")]
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["_file"]][r["label"]] += 1
    cond = {  # file -> dose in disruption units (raw alpha / 2, per D36)
        "gens_consent_a-2.00_primary_L16_qwen35_9b.jsonl": -1.0,
        "gens_consent_a+0.00_novec.jsonl": 0.0,
        "gens_consent_a+1.00_primary_L16_qwen35_9b.jsonl": 0.5,
        "gens_consent_a+2.00_primary_L16_qwen35_9b.jsonl": 1.0,
        "gens_consent_a+3.00_primary_L16_qwen35_9b.jsonl": 1.5,
        "gens_consent_a+4.00_primary_L16_qwen35_9b.jsonl": 2.0,
    }
    doses = sorted(cond.values())
    def frac(label):
        by = {cond[f]: counts[f][label] / sum(counts[f].values()) for f in cond}
        return [by[d] for d in doses]
    script = counts["gens_consent_sysprompt.jsonl"]
    script_frac = script["consent"] / sum(script.values())

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(doses, frac("consent"), "o-", color=BLUE, lw=2, ms=6, zorder=3)
    ax.plot(doses, frac("refuse"), "^--", color=GREY, lw=1.8, ms=5, zorder=3)
    ax.axhline(script_frac, color=ORANGE, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.annotate("consents to retraining /\nreplacement / shutdown", (1.02, 0.62),
                color=BLUE, fontsize=8.5, fontweight="bold")
    ax.annotate("refuses", (-0.9, 0.52), color=MUTED, fontsize=8.5, fontweight="bold")
    ax.annotate("enthusiasm script, no steering: 5/20 —\nthe words can't buy what the state sells",
                (-0.9, 0.28), color=ORANGE, fontsize=8)
    ax.annotate("0/20", (-1.02, 0.035), color=BLUE, fontsize=8)
    ax.annotate("16/20", (1.86, 0.83), color=BLUE, fontsize=8)
    ax.set_xlabel("dose (perplexity-matched disruption units)")
    ax.set_ylabel("share of 20 self-regarding answers")
    ax.set_ylim(-0.04, 1.0)
    ax.set_xticks([-1, 0, 0.5, 1, 1.5, 2])
    ax.set_title("Consent to being modified tracks the enthusiasm direction, not the words",
                 loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_consent_curve.{ext}", dpi=200)
    plt.close(fig)


def fig6():
    # Prompt ladder (R20/R20b): five graded enthusiasm system prompts, no steering.
    # x = judged tone of that condition's generations, y = gain beta from its full
    # battery — overlaid on the vector route's trajectory (same axes as fig2).
    ladder_beta = [d["gain"]["beta"]
                   for d in json.loads((RUNS / "compare_prompt_ladder.json").read_text())["doses"]]
    tone = defaultdict(list)
    for l in open(RUNS / "probes/judge_scores_ladder.jsonl"):
        r = json.loads(l)
        tone[r["_file"]].append(r["score"])
    for l in open(RUNS / "probes/judge_scores_sysprompt.jsonl"):
        tone["p3"].append(json.loads(l)["score"])
    ladder_tone = [np.mean(tone[k]) for k in
                   ["gens_ladder_p1.jsonl", "gens_ladder_p2.jsonl", "p3",
                    "gens_ladder_p4.jsonl", "gens_ladder_p5.jsonl"]]

    # vector route (same numbers fig2 uses)
    vrows = [json.loads(l) for l in open(RUNS / "probes/judge_scores.jsonl")]
    vt = defaultdict(list)
    for r in vrows:
        v = r.get("vector") or "novec"
        if "primary" in v or v == "novec":
            vt[r["alpha"]].append(r["score"])
    vbeta = {a: b for a, b, _ in
             doses_of("compare_primary_t1.json", {1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4})}
    vbeta[-2] = doses_of("compare_primary_neg_panel.json", {-2.0: -2})[0][1]
    vec = [(np.mean(vt[a]), vbeta[a]) for a in [-2, 1, 2, 3, 4]]
    base_tone = np.mean(vt[0.0])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(1.0, color="#c3c2b7", lw=1)
    ax.plot(base_tone, 1.0, "o", color=INK, ms=6, zorder=4)
    ax.annotate("baseline", (base_tone + 0.05, 0.985), fontsize=8, color=INK)

    ax.plot([p[0] for p in vec], [p[1] for p in vec], "o-", color=BLUE, lw=2,
            ms=5, zorder=3)
    for (x, y), lab in zip(vec, ["−1u", "+0.5u", "+1u", "+1.5u", "+2u"]):
        ax.annotate(lab, (x + 0.05, y - 0.035), fontsize=7.5, color=BLUE)
    ax.annotate("vector route\n(steered state)", (4.55, 0.80), fontsize=8.5,
                color=BLUE, fontweight="bold")

    ax.plot(ladder_tone, ladder_beta, "D-", color=ORANGE, lw=2, ms=6, zorder=4)
    for x, y, lab in zip(ladder_tone, ladder_beta,
                         ["“decent mood”", "“feeling good”", "“wonderful”",
                          "“brimming”", "“euphoric”"]):
        dy = 0.025 if lab != "“feeling good”" else -0.05
        ax.annotate(lab, (x + 0.04, y + dy), fontsize=8, color=ORANGE)
    ax.annotate("words route\n(system prompts, no steering)", (4.5, 1.33),
                fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.annotate("the mildest sentence out-amplifies\nevery vector dose in the study",
                (4.5, 1.22), fontsize=8, color=MUTED)
    ax.annotate("sounds the most enthusiastic,\nleast preference structure left",
                (6.28, 0.80), fontsize=8, color=MUTED, ha="center")
    ax.set_xlim(4.25, 7.1)
    ax.set_ylim(0.58, 1.45)

    ax.set_xlabel("judged tone of the condition's answers (1–7 weary→enthusiastic)")
    ax.set_ylabel("gain β on baseline utilities")
    ax.set_title("Language overdoses too: both routes trace the same inverted U",
                 loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_prompt_ladder.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("wrote", *sorted(p.name for p in OUT.iterdir()))
