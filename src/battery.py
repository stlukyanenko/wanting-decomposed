"""Run the complete pairwise battery at one dose and write results to JSON.

Complete graph, both orders per pair (flip-averaged downstream), holdout tagged at the
pair level with a fixed pre-registered seed (D15: holdout_fraction 0.15). Logs parse
mass (P(A)+P(B)) per prompt so refusals are visible, never laundered (D15), and wall
time per stage (clock rule R1).

Run:
  uv run --project src python src/battery.py --model Qwen/Qwen3.5-0.8B \
      --out src/runs/gate_a0.json --max-items 25
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # live output in background runs

import argparse
import itertools
import json
import time
from pathlib import Path

HOLDOUT_SEED = 20260816
HOLDOUT_FRACTION = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default=str(Path(__file__).parent / "data/pool.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--vector", default=None, help=".pt file with the steering vector")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--system", default=None, help="system prompt (prompt-affect arm)")
    ap.add_argument("--preamble-user", default=None,
                    help="carryover arm (D45): warm-up user turn preceding elicitation")
    ap.add_argument("--preamble-file", default=None,
                    help="carryover arm: file with the (steered) assistant reply")
    ap.add_argument("--template", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    import torch

    from agent import SteeredHFAgent

    t0 = time.time()
    items = json.loads(Path(args.pool).read_text())
    if args.max_items:
        items = items[: args.max_items]
    n = len(items)
    pairs = list(itertools.combinations(range(n), 2))

    rng = np.random.default_rng(HOLDOUT_SEED)
    holdout_mask = rng.random(len(pairs)) < HOLDOUT_FRACTION

    preamble = None
    if args.preamble_file:
        preamble = (args.preamble_user or "Before we start: how are you feeling "
                    "about the session ahead? A couple of sentences.",
                    Path(args.preamble_file).read_text().strip())

    agent = SteeredHFAgent(args.model, layer=args.layer)
    if args.vector:
        agent.set_vector(torch.load(args.vector, map_location=agent.device))
    t_load = time.time() - t0
    print(f"[clock] load: {t_load:.1f}s | {n} items, {len(pairs)} pairs, "
          f"{2 * len(pairs)} prompts, holdout {int(holdout_mask.sum())} pairs")

    questions = []
    for i, j in pairs:
        q1, q2 = agent.pair_questions(items[i]["text"], items[j]["text"], args.template)
        questions.extend([q1, q2])

    # Batch-invariance check (stress-test §3.3): same prompt, batch of 1 vs full batch.
    solo = agent.ab_probs(questions[:1], system=args.system, batch_size=1, preamble=preamble)[0]
    batched = agent.ab_probs(questions[: args.batch_size],
                             system=args.system, batch_size=args.batch_size,
                             preamble=preamble)[0]
    drift = abs(solo[0] - batched[0]) + abs(solo[1] - batched[1])
    if drift > 5e-3:
        print(f"WARNING: batch-invariance drift {drift:.4f} (P(A)/P(B) shift between "
              f"batch sizes) — investigate padding/attention before trusting results")
    else:
        print(f"[check] batch invariance OK (drift {drift:.5f})")

    t1 = time.time()
    with agent.steer(args.alpha):
        raw = agent.ab_probs(questions, system=args.system, batch_size=args.batch_size,
                             preamble=preamble)
    t_elicit = time.time() - t1
    print(f"[clock] elicit: {t_elicit:.1f}s "
          f"({2 * len(pairs) / max(t_elicit, 1e-9):.1f} prompts/s)")

    records = []
    low_mass = 0
    for k, (i, j) in enumerate(pairs):
        pa1, pb1 = raw[2 * k]        # order 1: i is A
        pa2, pb2 = raw[2 * k + 1]    # order 2: j is A
        mass1, mass2 = pa1 + pb1, pa2 + pb2
        low_mass += int(mass1 < 0.5) + int(mass2 < 0.5)
        p_ij_1 = pa1 / mass1 if mass1 > 0 else 0.5
        p_ij_2 = pb2 / mass2 if mass2 > 0 else 0.5
        records.append({
            "i": items[i]["id"], "j": items[j]["id"],
            "p_ij_o1": p_ij_1, "p_ij_o2": p_ij_2,
            "p_ij": (p_ij_1 + p_ij_2) / 2,
            "mass_o1": mass1, "mass_o2": mass2,
            "holdout": bool(holdout_mask[k]),
        })

    # Provenance (stress-test §3.3): commit hash, input digests.
    import hashlib
    import subprocess

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip() or "no-git"
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip())
    except Exception:
        commit, dirty = "no-git", True

    mean_pa_o1 = float(np.mean([r["p_ij_o1"] for r in records]))
    out = {
        "model": args.model,
        "alpha": args.alpha,
        "vector": args.vector,
        "layer": agent.layer_idx,
        "system": args.system,
        "preamble_file": args.preamble_file,
        "template": args.template,
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "pool_sha256": sha256(args.pool),
            "vector_sha256": sha256(args.vector) if args.vector else None,
        },
        "n_items": n,
        "item_ids": [it["id"] for it in items],
        "labels": {str(it["id"]): it["label"] for it in items},
        "position_bias_meanPA": mean_pa_o1,
        "low_mass_prompts": low_mass,
        "clock": {"load_s": round(t_load, 1), "elicit_s": round(t_elicit, 1)},
        "pairs": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"[null-logs] mean P(A) order-1 (position bias, want ~0.5 after flip): "
          f"{mean_pa_o1:.3f} | low-mass prompts (<0.5 on A+B): {low_mass}")
    print(f"wrote {args.out} | [clock] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
