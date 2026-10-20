"""Set-based extraction metrics with bootstrap confidence intervals."""

from __future__ import annotations

import random
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def doc_counts(pred: Set[str], gold: Set[str]) -> Tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def evaluate(preds: Sequence[Set[str]], golds: Sequence[Set[str]], invalid: Optional[Sequence[int]] = None,
             raw: Optional[Sequence[int]] = None, times: Optional[Sequence[float]] = None,
             n_boot: int = 1000, seed: int = 13) -> Dict[str, float]:
    """Micro / macro P-R-F1, exact match, invalid-label rate, timing and a bootstrap CI for micro-F1.

    Documents whose gold set AND prediction set are both empty count as exact matches and are
    excluded from macro averages (P/R undefined).
    """
    assert len(preds) == len(golds)
    counts = [doc_counts(p, g) for p, g in zip(preds, golds)]
    tp, fp, fn = (sum(c[i] for c in counts) for i in range(3))
    mp, mr, mf = prf(tp, fp, fn)
    per_doc = [prf(*c) for c, p, g in zip(counts, preds, golds) if p or g]
    out = {
        "n_docs": len(golds), "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "macro_precision": mean(x[0] for x in per_doc) if per_doc else 0.0,
        "macro_recall": mean(x[1] for x in per_doc) if per_doc else 0.0,
        "macro_f1": mean(x[2] for x in per_doc) if per_doc else 0.0,
        "exact_match": mean(1.0 if p == g else 0.0 for p, g in zip(preds, golds)) if golds else 0.0,
        "avg_predicted": mean(len(p) for p in preds) if preds else 0.0,
        "avg_gold": mean(len(g) for g in golds) if golds else 0.0,
    }
    if invalid is not None and raw is not None:
        total_raw = sum(raw)
        out["invalid_outputs"] = sum(invalid)
        out["invalid_label_rate"] = (sum(invalid) / total_raw) if total_raw else 0.0
    if times:
        out["time_mean_s"] = mean(times)
        out["time_median_s"] = median(times)
    lo, hi = bootstrap_f1(counts, n_boot, seed)
    out["micro_f1_ci95_low"], out["micro_f1_ci95_high"] = lo, hi
    return out


def bootstrap_f1(counts: List[Tuple[int, int, int]], n_boot: int = 1000, seed: int = 13) -> Tuple[float, float]:
    if not counts or n_boot <= 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(counts)
    vals = []
    for _ in range(n_boot):
        s = [counts[rng.randrange(n)] for _ in range(n)]
        vals.append(prf(sum(c[0] for c in s), sum(c[1] for c in s), sum(c[2] for c in s))[2])
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[min(n_boot - 1, int(0.975 * n_boot))]


def paired_bootstrap(counts_a: List[Tuple[int, int, int]], counts_b: List[Tuple[int, int, int]],
                     n_boot: int = 2000, seed: int = 13) -> Dict[str, float]:
    """Paired bootstrap of micro-F1(B) - micro-F1(A) over documents. Returns delta, 95% CI and p(delta<=0)."""
    assert len(counts_a) == len(counts_b)
    n = len(counts_a)
    if n == 0:
        return {"delta_f1": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value_one_sided": 1.0}
    f = lambda cs: prf(sum(c[0] for c in cs), sum(c[1] for c in cs), sum(c[2] for c in cs))[2]
    base = f(counts_b) - f(counts_a)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(f([counts_b[i] for i in idx]) - f([counts_a[i] for i in idx]))
    deltas.sort()
    return {"delta_f1": base, "ci95_low": deltas[int(0.025 * n_boot)],
            "ci95_high": deltas[min(n_boot - 1, int(0.975 * n_boot))],
            "p_value_one_sided": sum(1 for d in deltas if d <= 0) / n_boot}


def project(ids: Iterable[str], fn) -> Set[str]:
    """Map a prediction/gold set to a coarser level (e.g. occupation URI -> ISCO unit group)."""
    return {y for y in (fn(x) for x in ids) if y}
