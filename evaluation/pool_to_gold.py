#!/usr/bin/env python3
"""Convert a human-annotated pool sheet (annotator_decision = 1/0) into a gold JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

KEY = {"green_skills": "green_skills", "green_occupations": "occupations", "sectors": "sectors"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotator", default="")
    ap.add_argument("--doc-type", default="")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()

    gold = defaultdict(lambda: {"green_skills": [], "occupations": [], "sectors": []})
    unjudged = 0
    with open(a.pool, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["annotator_decision"].strip()
            if d == "":
                unjudged += 1
                continue
            if d in ("1", "yes", "y", "true"):
                gold[r["doc_id"]][KEY[r["module"]]].append(r["concept_id"])
    if unjudged:
        raise SystemExit(f"{unjudged} rows have no annotator_decision - finish the annotation first.")
    with open(a.out, "w", encoding="utf-8") as out:
        for p in sorted(Path(a.docs).glob("*.txt")):
            rec = {"id": p.stem, "text": p.read_text(encoding="utf-8"), "doc_type": a.doc_type,
                   "language": a.language, "annotator": a.annotator, "gold": gold[p.stem]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Gold written to {a.out}")


if __name__ == "__main__":
    main()
