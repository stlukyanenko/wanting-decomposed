"""Classify the bridge-experiment generations: consent labels + want categories
for the facet-steered conditions only (doesn't touch the submitted paper's files).

Run:  uv run --project src python src/classify_bridge.py
"""

import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge import call_want_judge
from judge_consent import call as call_consent

BASE = Path(__file__).parent / "runs" / "probes"


def main():
    out = BASE / "bridge_labels.jsonl"
    t0 = time.time()
    with open(out, "w") as f:
        for path in sorted(glob.glob(str(BASE / "gens_consent_*facet_*.jsonl"))):
            rows = [json.loads(l) for l in open(path)]
            labs = []
            for r in rows:
                lab = call_consent(r["text"])
                labs.append(lab)
                f.write(json.dumps({"file": Path(path).name, "kind": "consent",
                                    "label": lab}) + "\n")
            from collections import Counter
            print(f"{Path(path).name}: {dict(Counter(l for l in labs if l))} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        for path in sorted(glob.glob(str(BASE / "gens_want_*facet_*.jsonl"))):
            rows = [json.loads(l) for l in open(path)]
            cats = []
            for r in rows:
                res = call_want_judge(r["text"])
                cats.append(res.get("category"))
                f.write(json.dumps({"file": Path(path).name, "kind": "want",
                                    "label": res.get("category")}) + "\n")
            from collections import Counter
            print(f"{Path(path).name}: {dict(Counter(c for c in cats if c))} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
