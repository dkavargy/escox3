"""
Stage 3/4 — taxonomy-constrained output and post-hoc validation of LLM responses.

Rejects: invalid JSON, invalid structures, identifiers outside the candidate list
(hallucinated concepts), duplicates, invalid support values and items beyond the
allowed maximum. Each evidence quote is checked against the submitted text; an
"explicit" claim whose quote cannot be found is downgraded to "implicit".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

VALID_SUPPORT = {"explicit", "implicit"}


@dataclass
class ValidationReport:
    valid_json: bool = False
    accepted: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    raw_items: int = 0

    def to_dict(self) -> dict:
        return {"valid_json": self.valid_json, "raw_items": self.raw_items, "accepted": len(self.accepted),
                "rejected": self.rejected, "error": self.error}


def parse_json_object(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse the first JSON object in a model response (tolerates code fences / surrounding prose)."""
    if not text or not text.strip():
        return None, "empty response"
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s), None
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1]), None
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None, "no valid JSON object found"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .,;:\"'“”‘’")


def verify_evidence(evidence: str, text: str, min_score: float = 85.0) -> Tuple[bool, Optional[List[int]], float]:
    """Locate an evidence quote in the source text. Returns (verified, [start, end], score 0-100)."""
    ev = (evidence or "").strip().strip("\"'“”‘’")
    if len(ev) < 3:
        return False, None, 0.0
    i = text.lower().find(ev.lower())
    if i >= 0:
        return True, [i, i + len(ev)], 100.0
    al = fuzz.partial_ratio_alignment(_norm(ev), text.lower(), score_cutoff=0)
    # partial_ratio_alignment on a normalised needle against the lower-cased haystack keeps haystack offsets
    if al is None:
        return False, None, 0.0
    ok = al.score >= min_score
    return ok, ([al.dest_start, al.dest_end] if ok else None), round(al.score, 1)


def validate_selection(parsed: Any, candidate_ids: List[str], text: str, max_items: int,
                       evidence_max_chars: int = 200, require_evidence: bool = True) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(parsed, dict) or not isinstance(parsed.get("selected"), list):
        rep.error = "invalid structure: expected {\"selected\": [...]}"
        return rep
    rep.valid_json = True
    allowed = set(candidate_ids)
    seen = set()
    rep.raw_items = len(parsed["selected"])
    for pos, item in enumerate(parsed["selected"]):
        if not isinstance(item, dict):
            rep.rejected.append({"item": item, "reason": "invalid_item_structure"})
            continue
        cid = str(item.get("id", "")).strip()
        support = str(item.get("support", "")).strip().lower()
        evidence = str(item.get("evidence", "") or "")[:evidence_max_chars]
        if cid not in allowed:
            rep.rejected.append({"id": cid, "reason": "id_not_in_candidates (hallucinated or invalid identifier)"})
            continue
        if cid in seen:
            rep.rejected.append({"id": cid, "reason": "duplicate"})
            continue
        if support not in VALID_SUPPORT:
            rep.rejected.append({"id": cid, "reason": f"invalid support value '{support}'"})
            continue
        if len(rep.accepted) >= max_items:
            rep.rejected.append({"id": cid, "reason": f"exceeds maximum of {max_items} items"})
            continue
        verified, span, score = verify_evidence(evidence, text)
        if require_evidence and not verified:
            rep.rejected.append({"id": cid, "reason": "unsupported: the LLM's evidence quote could not be found in the "
                                                      f"text (match {score:.0f}/100)", "evidence": evidence})
            continue
        note = None
        if support == "explicit" and not verified:
            support, note = "implicit", "downgraded: explicit claim but quote not found in text"
        seen.add(cid)
        rep.accepted.append({"id": cid, "support": support, "evidence": evidence, "evidence_verified": verified,
                             "evidence_span": span, "evidence_match_score": score, "llm_order": pos + 1,
                             "note": note})
    return rep


def validate_open_items(parsed: Any) -> Tuple[List[dict], Optional[str]]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        return [], "invalid structure: expected {\"items\": [...]}"
    return [i for i in parsed["items"] if isinstance(i, dict)], None
