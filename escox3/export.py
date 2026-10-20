"""Export of analysis results (JSON / CSV) with human-readable labels and taxonomy identifiers."""

from __future__ import annotations

import csv
import io
import json
from typing import Dict, List

CSV_COLUMNS = [
    "category", "rank", "label", "esco_uri", "esco_code", "isco_code", "isco_label", "isco_major", "isco_sub_major",
    "isco_minor", "nace_code", "nace_level", "nace_hierarchy", "green", "green_status", "green_basis", "support",
    "confidence", "similarity", "llm_decision", "evidence", "evidence_verified", "candidate_source", "method",
]


def _isco_level(item: dict, level: str) -> str:
    for g in item.get("isco_hierarchy") or []:
        if g.get("level") == level:
            return f"{g['code']} {g.get('label') or ''}".strip()
    return ""


def result_rows(result: Dict) -> List[dict]:
    rows = []
    for item in result.get("green_skills") or []:
        rows.append({"category": "green_skill", "rank": item.get("rank"), "label": item.get("label"),
                     "esco_uri": item.get("esco_uri"), "green": True, "green_status": "green",
                     "green_basis": item.get("green_basis"), **_common(item)})
    for item in result.get("green_occupations") or []:
        gs = item.get("green_status") or {}
        rows.append({"category": "occupation", "rank": item.get("rank"), "label": item.get("label"),
                     "esco_uri": item.get("esco_uri"), "esco_code": item.get("esco_code"),
                     "isco_code": item.get("isco_code"), "isco_label": item.get("isco_label"),
                     "isco_major": _isco_level(item, "major"), "isco_sub_major": _isco_level(item, "sub_major"),
                     "isco_minor": _isco_level(item, "minor"), "green": {"green": True, "not_green": False}.get(gs.get("value")),
                     "green_status": gs.get("value"), "green_basis": gs.get("basis"), **_common(item)})
    for item in result.get("sectors") or []:
        hier = " > ".join(f"{h['code']} {h.get('title') or '(title not in supplied data)'}" for h in item.get("hierarchy") or [])
        rows.append({"category": "sector", "rank": item.get("rank"), "label": item.get("label"),
                     "nace_code": item.get("nace_code"), "nace_level": item.get("level"), "nace_hierarchy": hier,
                     **_common(item)})
    return rows


def _common(item: dict) -> dict:
    return {"support": item.get("support"), "confidence": item.get("confidence"), "similarity": item.get("similarity"),
            "llm_decision": item.get("llm_decision"), "evidence": item.get("evidence"),
            "evidence_verified": item.get("evidence_verified"),
            "candidate_source": "+".join(item.get("candidate_source") or []), "method": item.get("method")}


def to_csv(result: Dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in result_rows(result):
        w.writerow(r)
    for module, status in (result.get("module_status") or {}).items():
        if status.get("status") == "none_found":
            w.writerow({"category": module, "label": status.get("message")})
    return buf.getvalue()


def to_json(result: Dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)
