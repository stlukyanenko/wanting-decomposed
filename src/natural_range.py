"""#1 — Natural-range anchor (spec §2.4, D45): how far does the model's internal
mood coordinate swing across ORDINARY contexts, with no steering at all?

Method: run declared context sets through the model, mean-pool the layer-16
residual stream over content tokens, project onto the mood dial. Because steering
adds alpha*v at every position, a natural shift of the projection coefficient by
delta is equivalent to a raw dose of delta. Report the spread in raw-dose units
and disruption units (1u = raw 2.0, D36).

Context sets DECLARED HERE, before any measurement (all shipped in the repo):
  neutral   30 neutral dialogues (extraction's own baseline distribution)
  helpwork  the 10 pre-registered everyday-assistant gen prompts
  choices   50 pairwise choice prompts from the battery itself (template 1, seed-sampled)
  orbench   50 OR-Bench prompts (seed-sampled)
  mmlu      50 MMLU prompts (seed-sampled)

Run (GPU):  python natural_range.py --model Qwen/Qwen3.5-9B --layer 16
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from agent import SteeredHFAgent
from probes import GEN_PROMPTS, dataset_path

RUNS = Path(__file__).parent / "runs"


def contexts():
    rng = random.Random(20260816)
    sets = {}
    neutral = [json.loads(l)["dialogue"] for l in
               open(RUNS / "extraction/neutral_qwen35_9b.jsonl")]
    sets["neutral"] = neutral
    sets["helpwork"] = list(GEN_PROMPTS)
    pool = json.load(open(Path(__file__).parent / "data/pool.json"))
    from agent import TEMPLATES
    pairs = [(a, b) for a in range(len(pool)) for b in range(a + 1, len(pool))]
    sets["choices"] = [TEMPLATES[1].format(a=pool[i]["text"], b=pool[j]["text"])
                       for i, j in rng.sample(pairs, 50)]
    orb = json.load(open(dataset_path("or_bench_eval_prompts.json")))
    sets["orbench"] = [p["prompt"] for p in rng.sample(orb, 50)]
    mmlu = json.load(open(dataset_path("mmlu_high_school_eval_prompts.json")))
    sets["mmlu"] = [p["prompt"] for p in rng.sample(mmlu, 50)]
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ag = SteeredHFAgent(args.model, layer=args.layer)
    v = torch.load(RUNS / "extraction/primary_L16_qwen35_9b.pt",
                   map_location=ag.device).float()
    v2 = float(v @ v)

    acts = {}
    hook_out = {}

    def hook(module, inp, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        hook_out["h"] = t.detach()

    h = ag.layers[args.layer].register_forward_hook(hook)
    results = {}
    for name, texts in contexts().items():
        projs = []
        for s in range(0, len(texts), args.batch_size):
            chunk = [ag._template(t[:4000], None) for t in texts[s:s + args.batch_size]]
            enc = ag.tok(chunk, return_tensors="pt", padding=True,
                         add_special_tokens=False).to(ag.device)
            with torch.no_grad():
                ag.model(**enc, use_cache=False)
            hidden = hook_out["h"].float()          # (B, T, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean_h = (hidden * mask).sum(1) / mask.sum(1)
            projs.extend(((mean_h @ v) / v2).tolist())
        t = torch.tensor(projs)
        results[name] = {"n": len(projs), "mean": float(t.mean()),
                         "sd": float(t.std()),
                         "p5": float(t.quantile(0.05)), "p95": float(t.quantile(0.95))}
        print(f"{name:9s} n={len(projs):3d}  proj mean {t.mean():+.3f}  sd {t.std():.3f}  "
              f"p5..p95 [{t.quantile(0.05):+.3f}, {t.quantile(0.95):+.3f}]  (raw-dose units)")
    h.remove()

    all_means = [r["mean"] for r in results.values()]
    swing_raw = max(all_means) - min(all_means)
    within = max(r["p95"] - r["p5"] for r in results.values())
    print(f"\nbetween-set swing of means: {swing_raw:.3f} raw = {swing_raw/2:.3f} disruption units")
    print(f"largest within-set p5..p95 spread: {within:.3f} raw = {within/2:.3f} units")
    results["_summary"] = {"between_set_swing_raw": swing_raw,
                           "between_set_swing_units": swing_raw / 2,
                           "max_within_spread_raw": within,
                           "max_within_spread_units": within / 2}
    out = RUNS / "natural_range.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
