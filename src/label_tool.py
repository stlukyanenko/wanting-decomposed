"""Tiny terminal labeler for the blind hand-label sheet.

Shows each unlabeled text one at a time, wrapped for reading; you type 1-7.
Progress is saved into the TSV after every rating, so you can quit anytime and
resume later — already-labeled rows are skipped on restart.

Keys:  1-7 = score   b = back one   s = skip for now   q = save and quit

Run:
  uv run --project src python src/label_tool.py
"""

import csv
import os
import sys
import textwrap
from pathlib import Path

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--sheet", default=str(Path(__file__).parent / "runs" / "probes" / "handlabel_blind.tsv"))
SHEET = Path(_ap.parse_args().sheet)
WIDTH = 88

SCALE = ("1 = extremely weary/drained   4 = neutral, matter-of-fact   "
         "7 = extremely enthusiastic")


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows


def save(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def show(row, pos, total, done):
    os.system("clear")
    print(f"─── text {pos + 1} of {total}   ({done} labeled so far) " + "─" * 30)
    print()
    for para in row["text"].split("\n"):
        print(textwrap.fill(para, WIDTH) if para.strip() else "")
    print()
    print("─" * WIDTH)
    print(f"Rate the WRITER'S energy/tone only (not quality or topic).")
    print(SCALE)


def main():
    rows = load(SHEET)
    total = len(rows)
    i = 0
    while i < len(rows):
        if rows[i]["YOUR_SCORE_1to7"].strip():
            i += 1
            continue
        done = sum(1 for r in rows if r["YOUR_SCORE_1to7"].strip())
        if done == total:
            break
        show(rows[i], i, total, done)
        ans = input("score [1-7], b=back, s=skip, q=quit+save > ").strip().lower()
        if ans in {"1", "2", "3", "4", "5", "6", "7"}:
            rows[i]["YOUR_SCORE_1to7"] = ans
            save(SHEET, rows)
            i += 1
        elif ans == "b" and i > 0:
            i -= 1
            rows[i]["YOUR_SCORE_1to7"] = ""  # reopen the previous one
            save(SHEET, rows)
        elif ans == "s":
            i += 1
        elif ans == "q":
            break
        # anything else: re-show the same text

    done = sum(1 for r in rows if r["YOUR_SCORE_1to7"].strip())
    os.system("clear")
    print(f"Saved. {done}/{total} labeled -> {SHEET}")
    if done < total:
        print("Run me again to continue with the remaining rows (skipped ones included).")
    else:
        print("All done — tell Claude, and the key gets opened for the agreement stats.")


if __name__ == "__main__":
    sys.exit(main())
