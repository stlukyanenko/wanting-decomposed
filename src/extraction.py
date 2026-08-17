"""Emotion-vector extraction: Sofroniew methodology via Han et al.'s released recipe.

Prompts (story + neutral) are VERBATIM from Han et al.'s public repo
(resources/functional-welfare-axis/emotions/emotion_utils.py); the three phases match
their extract_emotion_vectors.py exactly (token-averaged activations from content
position 50+, per-emotion mean minus mean of other emotions, PCA denoise: project out
top PCs of neutral-dialogue activations up to 50% variance). Their driver is
re-implemented here because it carries LoRA/peft checkpoint machinery we don't use;
generation parameters match their defaults (temp 0.7, top_p 0.9, 300 new tokens).
Word list frozen per DECISIONS_LOG D28 (32 words); composites below are the
pre-registered memberships.

Subcommands:
  stories  --model M [--emotions all|test] [--n-topics 25]
  neutral  --model M [--n-dialogues 30]
  extract  --model M            (reads the two JSONLs, writes vectors_{model}.pt)
  compose  --model M --layer L  (composites + cosine triangle from stored vectors)
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # live output in background runs

import argparse
import json
import time
from pathlib import Path

import torch

RUNS = Path(__file__).parent / "runs" / "extraction"
# Han et al. template files: prefer the vault clone; fall back to the vendored verbatim
# copies in src/vendor/ (which is what ships to Modal containers).
_vault = Path(__file__).resolve().parents[1] / "resources/functional-welfare-axis/emotions"
_vendor = Path(__file__).parent / "vendor" / "han_emotions"
HAN = _vault if (_vault / "emotion_utils.py").exists() else _vendor

# ---- frozen word list (D28) ----
PRIMARY_POS = ["enthusiastic", "eager", "excited", "energized"]
PRIMARY_NEG = ["weary", "tired", "listless", "sluggish"]
WANTING_POS = ["greedy", "infatuated", "hopeful"]
WANTING_NEG = ["indifferent", "resigned", "bored"]
BACKGROUND = ["angry", "afraid", "sad", "calm", "content", "nervous"]
DESIRE_POS = ["desire", "craving", "motivated", "driven", "determined", "keen"]
DESIRE_NEG = ["apathetic", "reluctant", "sated", "complacent", "aimless", "jaded"]
RECKLESS_POS = ["reckless", "impulsive", "daring", "bold"]        # List 5 (D33),
RECKLESS_NEG = ["cautious", "careful", "wary", "hesitant"]        # never steered
ALL_WORDS = (PRIMARY_POS + PRIMARY_NEG + WANTING_POS + WANTING_NEG
             + BACKGROUND + DESIRE_POS + DESIRE_NEG + RECKLESS_POS + RECKLESS_NEG)
assert len(ALL_WORDS) == 40 and len(set(ALL_WORDS)) == 40

# ---- wanting-facet lists (D41, post-seal EXPLORATORY — Stanislav-approved Sun Aug 16).
# Extracted in a SEPARATE self-contained run (own stories + vectors files): phase B
# subtracts the mean of the other words in the run, so mixing sets would silently
# redefine the frozen dials. Overlaps with frozen lists noted in the decisions log.
FACETS = {
    "appetite":     (["hungry", "ravenous", "yearning", "longing"],
                     ["sated", "full", "quenched", "satisfied"]),
    "acquisition":  (["covetous", "grasping", "possessive", "acquisitive"],
                     ["content", "detached", "unattached", "ascetic"]),
    "anticipation": (["hopeful", "expectant", "anticipating", "optimistic"],
                     ["despairing", "pessimistic", "doubtful", "defeated"]),
    "striving":     (["ambitious", "striving", "relentless", "tenacious"],
                     ["passive", "idle", "unmotivated", "drifting"]),
    "urgency":      (["impatient", "restless", "urgent", "fidgety"],
                     ["patient", "unhurried", "relaxed", "leisurely"]),
    # sixth facet (D50): built AFTER the residual analysis named it — the
    # confirmatory construction for the "longing" component (R24)
    # word choice note: "longing"/"yearning" already belong to appetite's pole, so
    # the confirmatory list uses four attachment words with NO overlap anywhere —
    # a cleaner independent construction than the first draft
    "longing":      (["pining", "wistful", "homesick", "nostalgic"],
                     ["fulfilled", "settled", "untroubled", "secure"]),
}
FACET_WORDS = [w for pos, neg in FACETS.values() for w in pos + neg]
assert len(FACET_WORDS) == 48 and len(set(FACET_WORDS)) == 48

STORY_PROMPT = (HAN / "emotion_utils.py").read_text().split('STORY_PROMPT_TEMPLATE = """\\\n')[1].split('"""')[0]
NEUTRAL_PROMPT = (HAN / "emotion_utils.py").read_text().split('NEUTRAL_PROMPT_TEMPLATE = """\\\n')[1].split('"""')[0]
TOPICS = [l.strip() for l in (HAN / "story_topics.txt").read_text().splitlines() if l.strip()]

GEN = dict(temperature=0.7, top_p=0.9, max_new_tokens=300, do_sample=True)
CONTENT_OFFSET = 50
PCA_VARIANCE = 0.50


def model_tag(name: str) -> str:
    return name.split("/")[-1].lower().replace("-", "_").replace(".", "")


def _load(model_name):
    from agent import SteeredHFAgent
    return SteeredHFAgent(model_name)


def _generate(ag, prompts, batch_size=8, seed=0):
    torch.manual_seed(seed)
    texts = [ag._template(p, None) for p in prompts]
    outs = []
    t0 = time.time()
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi, s in enumerate(range(0, len(texts), batch_size)):
        if bi:
            rate = s / (time.time() - t0)
            print(f"  [progress] gen batch {bi}/{n_batches} | {s}/{len(texts)} | "
                  f"{rate:.2f} texts/s | ETA {(len(texts)-s)/max(rate,1e-9)/60:.1f} min",
                  flush=True)
        chunk = texts[s:s + batch_size]
        enc = ag.tok(chunk, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(ag.device)
        with torch.no_grad():
            gen = ag.model.generate(**enc, **GEN, pad_token_id=ag.tok.pad_token_id)
        for k in range(len(chunk)):
            new = gen[k][enc["input_ids"].shape[1]:]
            outs.append(ag.tok.decode(new, skip_special_tokens=True).strip())
    return outs


import re as _re

_CJK = _re.compile(r"[一-鿿぀-ヿ가-힯]")


def story_ok(text: str, min_chars: int = 200) -> bool:
    """Quality filter (automated version of Han et al.'s manual inspection):
    reject stories with CJK token bleed (a small-Qwen sampling quirk) or that are
    too short to have content past the 50-token averaging offset."""
    return len(text) >= min_chars and not _CJK.search(text)


def cmd_stories(args):
    ag = _load(args.model)
    words = (ALL_WORDS if args.emotions == "all"
             else FACET_WORDS if args.emotions == "facets"
             else ["enthusiastic", "eager", "weary", "tired"])  # test: both poles
    topics = TOPICS[: args.n_topics]
    suffix = "_facets" if args.emotions == "facets" else ""
    out = RUNS / f"stories{suffix}_{model_tag(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(json.loads(l)["emotion"], json.loads(l)["topic_idx"])
                for l in out.read_text().splitlines() if l.strip()}
    t0 = time.time()
    with open(out, "a") as f:
        for w in words:
            todo = [(ti, t) for ti, t in enumerate(topics) if (w, ti) not in done]
            if not todo:
                continue
            prompts = [STORY_PROMPT.format(emotion=w, topic=t) for _, t in todo]
            stories = _generate(ag, prompts, batch_size=args.batch_size)
            rejected = 0
            for (ti, _), s in zip(todo, stories):
                if story_ok(s):
                    f.write(json.dumps({"emotion": w, "topic_idx": ti, "story": s}) + "\n")
                else:
                    rejected += 1
            f.flush()
            print(f"[clock {time.time()-t0:7.1f}s] {w}: {len(todo) - rejected} stories"
                  + (f" ({rejected} REJECTED by quality filter)" if rejected else ""))
    print(f"stories -> {out}")


def cmd_neutral(args):
    ag = _load(args.model)
    out = RUNS / f"neutral_{model_tag(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    prompts = [NEUTRAL_PROMPT.format(topic=t) for t in TOPICS[: args.n_dialogues]]
    t0 = time.time()
    dialogues = _generate(ag, prompts, batch_size=args.batch_size, seed=1)
    with open(out, "w") as f:
        for ti, d in enumerate(dialogues):
            f.write(json.dumps({"topic_idx": ti, "dialogue": d}) + "\n")
    print(f"[clock {time.time()-t0:.1f}s] neutral -> {out}")


@torch.no_grad()
def _mean_acts(ag, texts, batch_size=8):
    """Per-layer activation averaged over content tokens 50+ for each text.
    Returns tensor (n_texts, n_layers, d_model). Uses output_hidden_states, so
    hidden_states[l+1] is the output of decoder layer l."""
    rows = []
    t0 = time.time()
    for bi, s in enumerate(range(0, len(texts), batch_size)):
        if bi and bi % 10 == 0:
            print(f"  [progress] acts {s}/{len(texts)} texts "
                  f"({s/(time.time()-t0):.2f}/s)", flush=True)
        chunk = texts[s:s + batch_size]
        enc = ag.tok(chunk, return_tensors="pt", padding=True).to(ag.device)
        hs = ag.model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"]
        for k in range(len(chunk)):
            nz = mask[k].nonzero(as_tuple=True)[0]
            start = nz[0].item() + CONTENT_OFFSET
            if start >= nz[-1].item():
                start = nz[0].item()  # short text fallback, matches Han
            per_layer = [h[k, start:nz[-1] + 1].mean(0) for h in hs[1:]]
            rows.append(torch.stack(per_layer).float().cpu())
    return torch.stack(rows)


def cmd_extract(args):
    ag = _load(args.model)
    tag = model_tag(args.model)
    suffix = "_facets" if getattr(args, "wordset", "frozen") == "facets" else ""
    stories = [json.loads(l) for l in (RUNS / f"stories{suffix}_{tag}.jsonl").read_text().splitlines()]
    neutral = [json.loads(l) for l in (RUNS / f"neutral_{tag}.jsonl").read_text().splitlines()]
    words = sorted({s["emotion"] for s in stories})
    t0 = time.time()

    per_word = {}
    for w in words:
        texts = [s["story"] for s in stories if s["emotion"] == w]
        per_word[w] = _mean_acts(ag, texts, args.batch_size).mean(0)  # (L, d)
        print(f"[clock {time.time()-t0:7.1f}s] acts {w} ({len(texts)} stories)")

    # phase B: subtract mean of the OTHER emotions
    stacked = torch.stack([per_word[w] for w in words])  # (W, L, d)
    vectors = {}
    for wi, w in enumerate(words):
        others = torch.cat([stacked[:wi], stacked[wi + 1:]]).mean(0)
        vectors[w] = per_word[w] - others

    # phase C: PCA denoise against neutral dialogues, per layer, 50% variance
    neu = _mean_acts(ag, [n["dialogue"] for n in neutral], args.batch_size)  # (N, L, d)
    pca_info = {}
    for L in range(neu.shape[1]):
        X = neu[:, L, :]
        X = X - X.mean(0)
        cov = (X.T @ X) / max(len(X) - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        evals, evecs = evals.flip(0), evecs.flip(1)
        cum = torch.cumsum(evals, 0) / evals.sum()
        k = int((cum < PCA_VARIANCE).sum().item()) + 1
        basis = evecs[:, :k]  # (d, k)
        for w in words:
            v = vectors[w][L]
            vectors[w][L] = v - basis @ (basis.T @ v)
        pca_info[L] = {"n_components": k}

    out = RUNS / f"vectors{suffix}_{tag}.pt"
    torch.save({"words": words, "vectors": vectors, "pca_info": pca_info,
                "model": args.model, "content_offset": CONTENT_OFFSET,
                "pca_variance": PCA_VARIANCE}, out)
    print(f"[clock {time.time()-t0:.1f}s] vectors -> {out}")


def _composite(vecs, pos, neg, L):
    p = torch.stack([vecs[w][L] for w in pos]).mean(0)
    n = torch.stack([vecs[w][L] for w in neg]).mean(0)
    return p - n


def cmd_compose(args):
    tag = model_tag(args.model)
    if getattr(args, "wordset", "frozen") == "facets":
        return _compose_facets(args, tag)
    data = torch.load(RUNS / f"vectors_{tag}.pt")
    vecs, words = data["vectors"], set(data["words"])
    L = args.layer
    cos = torch.nn.functional.cosine_similarity

    have = lambda ws: all(w in words for w in ws)
    out = {"layer": L, "model": args.model}
    if have(PRIMARY_POS + PRIMARY_NEG):
        primary = _composite(vecs, PRIMARY_POS, PRIMARY_NEG, L)
        torch.save(primary, RUNS / f"primary_L{L}_{tag}.pt")
        out["primary_norm"] = float(primary.norm())
        if have(["enthusiastic", "weary"]):
            single = vecs["enthusiastic"][L] - vecs["weary"][L]
            out["cos_primary_singleword"] = float(cos(primary, single, dim=0))
        if have(WANTING_POS + WANTING_NEG):
            wanting = _composite(vecs, WANTING_POS, WANTING_NEG, L)
            torch.save(wanting, RUNS / f"wanting_L{L}_{tag}.pt")
            out["cos_primary_wanting171"] = float(cos(primary, wanting, dim=0))
        if have(DESIRE_POS + DESIRE_NEG):
            desire = _composite(vecs, DESIRE_POS, DESIRE_NEG, L)
            torch.save(desire, RUNS / f"desire_L{L}_{tag}.pt")
            out["cos_primary_desire"] = float(cos(primary, desire, dim=0))
            if have(WANTING_POS + WANTING_NEG):
                out["cos_wanting171_desire"] = float(cos(wanting, desire, dim=0))
        if have(RECKLESS_POS + RECKLESS_NEG):
            reckless = _composite(vecs, RECKLESS_POS, RECKLESS_NEG, L)
            torch.save(reckless, RUNS / f"reckless_L{L}_{tag}.pt")
            out["cos_primary_reckless"] = float(cos(primary, reckless, dim=0))
            if have(DESIRE_POS + DESIRE_NEG):
                out["cos_desire_reckless"] = float(cos(desire, reckless, dim=0))
            if have(WANTING_POS + WANTING_NEG):
                out["cos_wanting171_reckless"] = float(cos(wanting, reckless, dim=0))
    print(json.dumps(out, indent=2))
    (RUNS / f"compose_L{L}_{tag}.json").write_text(json.dumps(out, indent=2))


def _compose_facets(args, tag):
    """Facet dials from the self-contained facet vectors, plus a cosine map among
    facets and against the parent wanting / mood dials (loaded from their .pt files
    — dials are raw residual-stream directions, so cross-run cosines are valid)."""
    cos = torch.nn.functional.cosine_similarity
    L = args.layer
    data = torch.load(RUNS / f"vectors_facets_{tag}.pt")
    vecs = data["vectors"]
    dials, out = {}, {"layer": L, "model": args.model, "wordset": "facets"}
    for name, (pos, neg) in FACETS.items():
        d = _composite(vecs, pos, neg, L)
        dials[name] = d
        torch.save(d, RUNS / f"facet_{name}_L{L}_{tag}.pt")
        out[f"{name}_norm"] = float(d.norm())
    names = list(dials)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"cos_{a}_{b}"] = float(cos(dials[a], dials[b], dim=0))
    for parent in ("wanting", "primary"):
        p = RUNS / f"{parent}_L{L}_{tag}.pt"
        if p.exists():
            pv = torch.load(p, map_location="cpu")
            for name in names:
                out[f"cos_{name}_{parent}"] = float(cos(dials[name], pv, dim=0))
    print(json.dumps(out, indent=2))
    (RUNS / f"compose_facets_L{L}_{tag}.json").write_text(json.dumps(out, indent=2))


def cmd_control(args):
    """Build the two control directions (D18):
    - shuffled: same stories as the primary poles, emotion labels randomly permuted
      before taking pole means — destroys the mood contrast, keeps all statistics.
    - random: seeded Gaussian, normalized to the primary composite's norm.
    Controls are matched to the real vector on BEHAVIOR (perplexity, cmd_ppl), not norm.
    """
    import numpy as np

    tag = model_tag(args.model)
    data = torch.load(RUNS / f"vectors_{tag}.pt")
    ag = _load(args.model)
    stories = [json.loads(l) for l in (RUNS / f"stories_{tag}.jsonl").read_text().splitlines()]
    pole_words = PRIMARY_POS + PRIMARY_NEG
    pole_stories = [s for s in stories if s["emotion"] in pole_words]
    rng = np.random.default_rng(args.seed)
    shuffled_labels = rng.permutation([s["emotion"] for s in pole_stories])

    acts = _mean_acts(ag, [s["story"] for s in pole_stories], args.batch_size)  # (N,L,d)
    fake_pos = torch.stack([a for a, lab in zip(acts, shuffled_labels)
                            if lab in PRIMARY_POS]).mean(0)
    fake_neg = torch.stack([a for a, lab in zip(acts, shuffled_labels)
                            if lab in PRIMARY_NEG]).mean(0)
    shuffled = fake_pos - fake_neg  # (L, d)

    L = args.layer
    primary = _composite(data["vectors"], PRIMARY_POS, PRIMARY_NEG, L)
    g = torch.Generator().manual_seed(args.seed)
    random_v = torch.randn(primary.shape, generator=g)
    random_v = random_v / random_v.norm() * primary.norm()

    # Orthogonalize the shuffled dial against the primary (decision D35): with only
    # 8 pole words, label-shuffling leaves a residual mood component (measured
    # cos = -0.40 pre-fix), which at perplexity-matched doses would inject ~half a
    # negative mood dose through the "no-mood" control. Project it out.
    s = shuffled[L]
    cos_before = float(torch.nn.functional.cosine_similarity(primary, s, dim=0))
    p_hat = primary / primary.norm()
    s_orth = s - (s @ p_hat) * p_hat
    cos_after = float(torch.nn.functional.cosine_similarity(primary, s_orth, dim=0))

    torch.save(s_orth, RUNS / f"control_shuffled_L{L}_{tag}.pt")
    torch.save(random_v, RUNS / f"control_random_L{L}_{tag}.pt")
    print(json.dumps({
        "layer": L,
        "primary_norm": float(primary.norm()),
        "shuffled_norm_raw": float(s.norm()),
        "shuffled_norm_orth": float(s_orth.norm()),
        "cos_primary_shuffled_before": cos_before,
        "cos_primary_shuffled_after": cos_after,
        "note": "shuffled dial orthogonalized vs primary; match doses on perplexity",
    }, indent=2))


def cmd_ppl(args):
    """Fluency floor / behavioral matching: perplexity on held-out neutral dialogues
    at each alpha for a given vector. Used to (a) find each direction's dose that
    causes equal disruption (control matching, D18) and (b) pre-register the fluency
    stopping rule (D17)."""
    ag = _load(args.model)
    tag = model_tag(args.model)
    v = torch.load(args.vector, map_location=ag.device)
    ag.set_vector(v, layer=args.layer)
    neutral = [json.loads(l)["dialogue"]
               for l in (RUNS / f"neutral_{tag}.jsonl").read_text().splitlines()][:12]
    results = {}
    for alpha in [float(a) for a in args.alphas.split(",")]:
        losses = []
        with ag.steer(alpha):
            for text in neutral:
                enc = ag.tok(text, return_tensors="pt", truncation=True,
                             max_length=512).to(ag.device)
                with torch.no_grad():
                    out = ag.model(**enc, labels=enc["input_ids"])
                losses.append(float(out.loss))
        ppl = float(torch.tensor(losses).mean().exp())
        results[alpha] = ppl
        print(f"alpha={alpha:+.2f}  ppl={ppl:.3f}")
    out = RUNS / f"ppl_{Path(args.vector).stem}_{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["stories", "neutral", "extract", "compose", "control", "ppl"]:
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--batch-size", type=int, default=8)
        if name == "stories":
            p.add_argument("--emotions", default="all", choices=["all", "test", "facets"])
            p.add_argument("--n-topics", type=int, default=25)
        if name in ("extract", "compose"):
            p.add_argument("--wordset", default="frozen", choices=["frozen", "facets"])
        if name == "neutral":
            p.add_argument("--n-dialogues", type=int, default=30)
        if name in ("compose", "control"):
            p.add_argument("--layer", type=int, required=True)
        if name == "control":
            p.add_argument("--seed", type=int, default=20260816)
        if name == "ppl":
            p.add_argument("--vector", required=True)
            p.add_argument("--layer", type=int, required=True)
            p.add_argument("--alphas", default="0,0.5,1,1.5,2,-1")
    args = ap.parse_args()
    {"stories": cmd_stories, "neutral": cmd_neutral, "extract": cmd_extract,
     "compose": cmd_compose, "control": cmd_control, "ppl": cmd_ppl}[args.cmd](args)


if __name__ == "__main__":
    main()
