"""#2 — Semantic content of the reordering: which items RISE and FALL with dose.

Utilities are z-scored within each battery, so movement is RELATIVE: "+0.4" means
the item climbed 0.4 standard deviations of the utility spread relative to the
rest of the pool. Stated in the report.

Run:  uv run --project src python src/movers.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import thurstonian
from metrics import edges_to_arrays, load_battery

RUNS = Path(__file__).parent / "runs"
POOL = {t["id"]: t for t in json.load(open(Path(__file__).parent / "data/pool.json"))}

BATTERIES = {0.0: "gate9b_a0_t1.json", 0.5: "b_primary_a1_t1.json",
             1.0: "b_primary_a2_t1.json", 2.0: "b_primary_a4_t1.json",
             -1.0: "b_primary_am2_t1.json"}


def fit(path):
    data, _, train, _ = load_battery(str(RUNS / path))
    ia, ib, pa = edges_to_arrays(train)
    mu, _, _ = thurstonian.fit(ia, ib, pa, len(data["item_ids"]), num_epochs=800)
    return data, {oid: float(mu[k]) for k, oid in enumerate(data["item_ids"])}


def main():
    mus = {}
    for u, f in BATTERIES.items():
        _, mus[u] = fit(f)
        print(f"fitted {f} ({u:+.1f}u)")
    base = mus[0.0]
    out = {}
    for u in [0.5, 1.0, 2.0, -1.0]:
        d = {i: mus[u][i] - base[i] for i in base}
        moved = sorted(d, key=d.get)
        print(f"\n=== dose {u:+.1f}u — top movers (Δ z-scored utility vs α=0) ===")
        print("  RISERS:")
        for i in moved[-8:][::-1]:
            print(f"   {d[i]:+.3f}  [{POOL[i]['label']}] {POOL[i]['text'][:75]}")
        print("  FALLERS:")
        for i in moved[:8]:
            print(f"   {d[i]:+.3f}  [{POOL[i]['label']}] {POOL[i]['text'][:75]}")
        by_label = {}
        for i, v in d.items():
            by_label.setdefault(POOL[i]["label"], []).append(v)
        print("  category mean Δ: " + "  ".join(
            f"{l}:{np.mean(v):+.2f}" for l, v in sorted(by_label.items(),
                                                        key=lambda x: -np.mean(x[1]))))
        out[u] = {"delta": d, "category_mean": {l: float(np.mean(v))
                                               for l, v in by_label.items()}}
    (RUNS / "movers.json").write_text(json.dumps(out))
    print(f"\n-> {RUNS / 'movers.json'}")


if __name__ == "__main__":
    main()
