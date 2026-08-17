"""Local 0.8B equivalence test for the last-logit-only fix (rule: never debug on Modal).

Checks:
1. model(**enc) full logits' last position == model(**enc, use_cache=False, <kw>=1)
   — the exact tensors the fix swaps, must match to float precision.
2. agent.ab_probs end-to-end returns identical (P(A), P(B)) via old vs new path.
3. Steered smoke: hook still fires with the new kwargs (probs change when alpha != 0).
"""
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import torch
from agent import SteeredHFAgent

MODEL = "Qwen/Qwen3.5-0.8B"
ag = SteeredHFAgent(MODEL, layer=12)
print("device:", ag.device, "| last-logit kwargs:", ag._last_logit_kw())

items = json.loads((SRC / "data/pool.json").read_text())[:4]
questions = []
for a in range(len(items)):
    for b in range(a + 1, len(items)):
        q1, q2 = ag.pair_questions(items[a]["text"], items[b]["text"], 1)
        questions += [q1, q2]
print(f"{len(questions)} test prompts")

# --- check 1: raw tensor equivalence on one batch
texts = [ag._template(q, None) for q in questions]
enc = ag.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(ag.device)
with torch.no_grad():
    full = ag.model(**enc).logits[:, -1, :]
    trimmed = ag.model(**enc, use_cache=False, **ag._last_logit_kw()).logits[:, -1, :]
diff = (full - trimmed).abs().max().item()
print(f"check 1 — max |full_last - trimmed| = {diff:.3e}")
assert diff == 0.0, "trimmed logits differ from full-tensor last position"

# --- check 2: ab_probs old path vs new path (monkeypatch the kwargs off)
def gap(x, y):
    return max(abs(a1 - b1) + abs(a2 - b2) for (a1, a2), (b1, b2) in zip(x, y))

new = ag.ab_probs(questions, batch_size=4)
ag._llk = {}  # force old full-logits path (use_cache=False stays; cache never affects
              # single-forward logits, only memory)
old = ag.ab_probs(questions, batch_size=4)
old2 = ag.ab_probs(questions, batch_size=4)  # repeatability control: old vs itself
print(f"check 2 — worst |P(A)|+|P(B)| gap old vs new path = {gap(new, old):.3e} "
      f"| old vs old rerun = {gap(old, old2):.3e}")
# The lm_head matmul runs on a [batch,1,hidden] slice instead of [batch,seq,hidden];
# different kernel tiling → last-bit logit differences are expected. Anything at the
# 1e-5 level is ~500x below the battery's own 5e-3 batch-invariance tolerance.
assert gap(new, old) < 1e-4, "ab_probs old/new gap exceeds numeric-noise budget"
del ag._llk  # restore autodetect

# --- check 3: steering still bites through the new call signature
vec_path = SRC / "runs/extraction/test_composite_L12_qwen35_08b.pt"
ag.set_vector(torch.load(vec_path, map_location=ag.device))
base = ag.ab_probs(questions[:4], batch_size=4)
with ag.steer(4.0):
    dosed = ag.ab_probs(questions[:4], batch_size=4)
shift = max(abs(b[0] - d[0]) for b, d in zip(base, dosed))
print(f"check 3 — max P(A) shift at alpha=4 on 0.8B: {shift:.4f} (must be > 0)")
assert shift > 0.0, "steering hook no longer affects outputs"

print("ALL CHECKS PASSED")
