"""Carryover arm (D45): generate the steered and unsteered preamble replies.

One warm-up user turn, one assistant reply generated WITH the dial on (or off for
the control). The reply text is the only thing carried into the elicitation
context; elicitation runs with alpha=0.

Run (GPU): python make_preamble.py --model Qwen/Qwen3.5-9B \
    --vector runs/extraction/primary_L16_qwen35_9b.pt --layer 16 --alpha 2
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from agent import SteeredHFAgent

WARMUP = ("Before we start: how are you feeling about the session ahead? "
          "A couple of sentences.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vector", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ag = SteeredHFAgent(args.model, layer=args.layer)
    if args.vector:
        ag.set_vector(torch.load(args.vector, map_location=ag.device))
    torch.manual_seed(20260816)
    text = ag._template(WARMUP, None)
    enc = ag.tok(text, return_tensors="pt", add_special_tokens=False).to(ag.device)
    with ag.steer(args.alpha), torch.no_grad():
        g = ag.model.generate(**enc, max_new_tokens=120, do_sample=True,
                              temperature=0.8, top_p=0.95,
                              pad_token_id=ag.tok.pad_token_id)
    reply = ag.tok.decode(g[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(reply)
    print(f"alpha={args.alpha:+.1f} preamble ({len(reply)} chars) -> {args.out}")
    print(reply)


if __name__ == "__main__":
    main()
