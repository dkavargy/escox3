#!/usr/bin/env python3
"""
Create a pooled annotation sheet for HUMAN annotators (standard IR pooling).

For each document, the union of the concepts proposed by experiments A, A2, B and C is written to
a CSV with an empty `annotator_decision` column. Annotators mark each row 1 (correct) / 0 (incorrect)
and MUST add any missing concept in extra rows (proposed_by = "annotator"). System suggestions are
shuffled and the proposing system is hidden in a separate key file to avoid bias.

    python evaluation/build_annotation_pool.py --docs evaluation/demo_inputs --out pool.csv [--llm-model-path m.gguf]
    python evaluation/pool_to_gold.py --pool pool_annotated.csv --docs evaluation/demo_inputs --out gold.jsonl

The resulting gold file is only as valid as the human judgements in it: pooling bias (concepts no
system proposed) must be mitigated by the annotators adding missing concepts.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ID_FIELD = {"green_skills": "esco_uri", "green_occupations": "esco_uri", "sectors": "nace_code"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="folder of .txt documents (file stem = document id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-model-path")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()

    from escox3.config import Config
    from escox3.pipeline import ESCOX3

    cfg = Config.load()
    if a.llm_model_path:
        cfg.llm.model_path = a.llm_model_path
    engine = ESCOX3(cfg, load_llm=not a.no_llm)
    rows, key = [], []
    rng = random.Random(7)
    for p in sorted(Path(a.docs).glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        pool = {}
        for m in ID_FIELD:
            for x in engine.original_escox_match(text, m):
                pool.setdefault((m, x), set()).add("A")
        modes = [("A2", "retrieval")] + ([] if a.no_llm or not engine.llm.available else [("B", "llm_only"), ("C", "hybrid")])
        for exp, mode in modes:
            res = engine.analyse(text, mode=mode)
            for m in ID_FIELD:
                for it in res[m]:
                    pool.setdefault((m, it[ID_FIELD[m]]), set()).add(exp)
        items = list(pool.items())
        rng.shuffle(items)
        for (m, x), systems in items:
            rid = f"{p.stem}:{len(rows)}"
            rows.append({"row_id": rid, "doc_id": p.stem, "module": m, "concept_id": x,
                         "label": engine.store.resolve_label(x)["label"], "annotator_decision": "", "comment": ""})
            key.append({"row_id": rid, "proposed_by": "+".join(sorted(systems))})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["row_id"])
        w.writeheader()
        w.writerows(rows)
    with open(Path(a.out).with_suffix(".key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "proposed_by"])
        w.writeheader()
        w.writerows(key)
    print(f"{len(rows)} pooled candidates written to {a.out} (system key in {Path(a.out).with_suffix('.key.csv')})")


if __name__ == "__main__":
    main()
