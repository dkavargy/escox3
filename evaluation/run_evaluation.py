#!/usr/bin/env python3
"""
ESCOX 3.0 evaluation framework.

Experiments (per module: green skills, occupations, sectors — evaluated separately):
  A   original ESCOX matching algorithm, unchanged (fragment split, max-similarity, fixed threshold),
      applied to the ESCOX 3.0 reference sets — "semantic retrieval only" baseline
  A2  ESCOX 3.0 retrieval stage only (improved fragmenting, top-k, thresholds) — ablation without LLM
  B   LLM only: the instruction-following LLM extracts labels without retrieval; outputs are mapped to the
      taxonomy by exact label/code lookup (non-matching outputs count as invalid / hallucinated)
  C   ESCOX 3.0 hybrid: retrieval + LLM validation + taxonomy-constrained decoding + validation

Gold data: JSONL produced by HUMAN annotators (see ANNOTATION_GUIDELINES.md and gold/TEMPLATE.jsonl).
Model outputs are never used as ground truth.

Usage:
  python evaluation/run_evaluation.py --gold evaluation/gold/my_gold.jsonl --experiments A,A2,B,C \
      --out evaluation/results/run1 [--llm-model-path model.gguf] [--sweep]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import doc_counts, evaluate, paired_bootstrap, project  # noqa: E402

MODULES = ["green_skills", "green_occupations", "sectors"]
GOLD_KEYS = {"green_skills": "green_skills", "green_occupations": "occupations", "sectors": "sectors"}
ID_FIELD = {"green_skills": "esco_uri", "green_occupations": "esco_uri", "sectors": "nace_code"}


def load_gold(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            if d.get("example_only"):
                continue
            if "text" not in d or "gold" not in d:
                raise ValueError(f"{path}:{n}: each record needs 'text' and 'gold'")
            docs.append(d)
    if not docs:
        raise SystemExit("No annotated documents found (records marked example_only are skipped). "
                         "Human-annotated gold data are required to compute metrics.")
    return docs


def validate_gold(docs, store):
    problems = []
    for d in docs:
        g = d["gold"]
        for u in g.get("green_skills", []):
            if store.skill(u) is None:
                problems.append(f"{d['id']}: unknown green skill {u}")
        for u in g.get("occupations", []):
            if store.occupation(u) is None:
                problems.append(f"{d['id']}: unknown occupation {u}")
        for c in g.get("sectors", []):
            if store.nace_entry(c) is None:
                problems.append(f"{d['id']}: unknown NACE code {c}")
    return problems


def run_experiment(engine, exp, doc, modules):
    """Return {module: (pred_set, n_invalid, n_raw_outputs)}, seconds."""
    t0 = time.time()
    out = {}
    if exp == "A":
        for m in modules:
            ids = set(engine.original_escox_match(doc["text"], m))
            out[m] = (ids, 0, len(ids))
    else:
        mode = {"A2": "retrieval", "B": "llm_only", "C": "hybrid"}[exp]
        res = engine.analyse(doc["text"], "green_skills" in modules, "green_occupations" in modules,
                             "sectors" in modules, mode=mode)
        if res["metadata"]["mode"] != mode:
            raise RuntimeError(f"Experiment {exp} requires an LLM but the engine fell back to "
                               f"{res['metadata']['mode']}: {res['metadata']['warnings']}")
        calls = {c["module"]: c for c in res["metadata"]["llm_calls"]}
        for m in modules:
            ids = {it[ID_FIELD[m]] for it in res[m]}
            invalid, raw = 0, len(ids)
            c = calls.get(m)
            if c and c.get("invoked"):
                v = c.get("validation") or {}
                if exp == "B":
                    invalid = v.get("invalid_labels", 0)
                    raw = v.get("raw_items", 0)
                else:
                    invalid = sum(1 for r in v.get("rejected", []) if "id_not_in_candidates" in r.get("reason", ""))
                    raw = v.get("raw_items", 0)
                if not c.get("valid_json"):
                    invalid += 1
                    raw += 1
            out[m] = (ids, invalid, raw)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--experiments", default="A,A2,B,C")
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--config", help="ESCOX TOML config")
    ap.add_argument("--embedding-model")
    ap.add_argument("--llm-backend")
    ap.add_argument("--llm-model-path")
    ap.add_argument("--sweep", action="store_true", help="threshold / top-k sensitivity analysis (A2 grid; C on a small grid)")
    ap.add_argument("--sweep-hybrid", action="store_true", help="also sweep the hybrid pipeline (slow: LLM calls)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    from escox3.config import Config
    from escox3.pipeline import ESCOX3

    cfg = Config.load(a.config)
    if a.embedding_model:
        cfg.embedding_model = a.embedding_model
    if a.llm_backend:
        cfg.llm.backend = a.llm_backend
    if a.llm_model_path:
        cfg.llm.model_path = a.llm_model_path
    exps = [e.strip() for e in a.experiments.split(",") if e.strip()]
    modules = [m.strip() for m in a.modules.split(",") if m.strip()]
    needs_llm = any(e in ("B", "C") for e in exps)
    engine = ESCOX3(cfg, load_llm=needs_llm, load_legacy=True)
    docs = load_gold(a.gold)
    problems = validate_gold(docs, engine.store)
    if problems:
        print("Gold file references unknown concepts:\n  " + "\n  ".join(problems[:30]))
        raise SystemExit(1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_doc_rows, err_rows, metric_rows = [], [], []
    preds = defaultdict(lambda: defaultdict(list))  # exp -> module -> list of (set, invalid, raw)
    times = defaultdict(list)

    for exp in exps:
        for i, d in enumerate(docs, 1):
            res, sec = run_experiment(engine, exp, d, modules)
            times[exp].append(sec)
            for m in modules:
                pset, inv, raw = res[m]
                gset = set(d["gold"].get(GOLD_KEYS[m], []))
                preds[exp][m].append((pset, inv, raw))
                tp, fp, fn = doc_counts(pset, gset)
                per_doc_rows.append({"experiment": exp, "doc_id": d["id"], "doc_type": d.get("doc_type", ""),
                                     "language": d.get("language", ""), "module": m, "tp": tp, "fp": fp, "fn": fn,
                                     "invalid": inv, "seconds": round(sec, 3)})
                lab = lambda x: engine.store.resolve_label(x)["label"] or x
                for x in sorted(pset - gset):
                    err_rows.append({"experiment": exp, "doc_id": d["id"], "module": m, "type": "FP", "id": x, "label": lab(x)})
                for x in sorted(gset - pset):
                    err_rows.append({"experiment": exp, "doc_id": d["id"], "module": m, "type": "FN", "id": x, "label": lab(x)})
            print(f"[{exp}] {i}/{len(docs)} {d['id']} {sec:.1f}s", flush=True)

    occ_to_isco = lambda u: (engine.store.occupation(u) or {}).get("isco_code")
    to_division = lambda c: c[:3] if len(c) >= 3 else None
    groups = {"all": lambda d: True}
    for key in ("doc_type", "language"):
        for v in sorted({d.get(key, "") for d in docs if d.get(key)}):
            groups[f"{key}={v}"] = (lambda k, val: (lambda d: d.get(k) == val))(key, v)

    for exp in exps:
        for m in modules:
            for gname, gf in groups.items():
                idx = [j for j, d in enumerate(docs) if gf(d)]
                if not idx:
                    continue
                P = [preds[exp][m][j][0] for j in idx]
                G = [set(docs[j]["gold"].get(GOLD_KEYS[m], [])) for j in idx]
                inv = [preds[exp][m][j][1] for j in idx]
                raw = [preds[exp][m][j][2] for j in idx]
                tms = [times[exp][j] for j in idx]
                variants = {"exact": (P, G)}
                if m == "green_occupations":
                    variants["isco_unit"] = ([project(p, occ_to_isco) for p in P], [project(g, occ_to_isco) for g in G])
                if m == "sectors":
                    variants["nace_division"] = ([project(p, to_division) for p in P], [project(g, to_division) for g in G])
                for vname, (pp, gg) in variants.items():
                    r = evaluate(pp, gg, inv, raw, tms, n_boot=a.n_boot)
                    metric_rows.append({"experiment": exp, "module": m, "level": vname, "group": gname, **r})

    # paired comparisons against the baseline
    comparisons = []
    for m in modules:
        G = [set(d["gold"].get(GOLD_KEYS[m], [])) for d in docs]
        for base, other in (("A", "C"), ("A2", "C"), ("B", "C"), ("A", "B")):
            if base in exps and other in exps:
                ca = [doc_counts(p[0], g) for p, g in zip(preds[base][m], G)]
                cb = [doc_counts(p[0], g) for p, g in zip(preds[other][m], G)]
                comparisons.append({"module": m, "baseline": base, "system": other, **paired_bootstrap(ca, cb)})

    # sensitivity analysis
    sweep_rows = []
    if a.sweep:
        r = engine.cfg.retrieval
        grids = {"green_skills": ("skill_accept_threshold", "skill_top_k_per_chunk"),
                 "green_occupations": ("occupation_accept_threshold", "occupation_top_k_per_chunk"),
                 "sectors": ("sector_accept_threshold", "sector_top_k_per_chunk")}
        for m in modules:
            thr_attr, k_attr = grids[m]
            orig = (getattr(r, thr_attr), getattr(r, k_attr))
            for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
                for k in (1, 3, 5, 10):
                    setattr(r, thr_attr, thr)
                    setattr(r, k_attr, k)
                    P, G = [], []
                    for d in docs:
                        res = engine.analyse(d["text"], m == "green_skills", m == "green_occupations", m == "sectors", mode="retrieval")
                        P.append({it[ID_FIELD[m]] for it in res[m]})
                        G.append(set(d["gold"].get(GOLD_KEYS[m], [])))
                    sweep_rows.append({"experiment": "A2", "module": m, "accept_threshold": thr, "top_k": k,
                                       **{k2: v for k2, v in evaluate(P, G, n_boot=0).items() if "ci95" not in k2}})
            setattr(r, thr_attr, orig[0])
            setattr(r, k_attr, orig[1])
        if a.sweep_hybrid and engine.llm.available:
            cand = {"green_skills": ("skill_candidate_threshold", "skill_max_candidates"),
                    "green_occupations": ("occupation_candidate_threshold", "occupation_max_candidates"),
                    "sectors": ("sector_candidate_threshold", "sector_max_candidates")}
            for m in modules:
                thr_attr, n_attr = cand[m]
                orig = (getattr(r, thr_attr), getattr(r, n_attr))
                for thr in (0.30, 0.40, 0.50):
                    for n in (5, 10, 20):
                        setattr(r, thr_attr, thr)
                        setattr(r, n_attr, n)
                        P, G = [], []
                        for d in docs:
                            res = engine.analyse(d["text"], m == "green_skills", m == "green_occupations", m == "sectors", mode="hybrid")
                            P.append({it[ID_FIELD[m]] for it in res[m]})
                            G.append(set(d["gold"].get(GOLD_KEYS[m], [])))
                        sweep_rows.append({"experiment": "C", "module": m, "candidate_threshold": thr, "max_candidates": n,
                                           **{k2: v for k2, v in evaluate(P, G, n_boot=0).items() if "ci95" not in k2}})
                setattr(r, thr_attr, orig[0])
                setattr(r, n_attr, orig[1])

    def write_csv(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv("metrics.csv", metric_rows)
    write_csv("per_document.csv", per_doc_rows)
    write_csv("errors.csv", err_rows)
    write_csv("comparisons.csv", comparisons)
    write_csv("sensitivity.csv", sweep_rows)
    manifest = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "gold_file": str(a.gold),
                "n_docs": len(docs), "experiments": exps, "modules": modules, "config": engine.cfg.to_dict(),
                "llm": engine.llm.info(), "taxonomy": engine.store.manifest}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults written to {out}")
    for r in metric_rows:
        if r["group"] == "all":
            print(f"{r['experiment']:>3} {r['module']:<18} {r['level']:<13} P={r['micro_precision']:.3f} "
                  f"R={r['micro_recall']:.3f} F1={r['micro_f1']:.3f} [{r['micro_f1_ci95_low']:.3f},{r['micro_f1_ci95_high']:.3f}] "
                  f"EM={r['exact_match']:.3f} invalid={r.get('invalid_label_rate', 0):.3f} t={r.get('time_mean_s', 0):.2f}s")


if __name__ == "__main__":
    main()
