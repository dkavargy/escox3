"""
Reference taxonomy store and identifier -> label mapping layer.

Every concept returned by ESCOX 3.0 is looked up here. Identifiers (ESCO URIs, ISCO-08
codes, NACE codes) are kept internally for traceability, while the GUI shows the exact
preferred labels from the supplied files. Unknown identifiers are reported as
*unresolved* — a label is never invented.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import REFERENCE_DIR
from . import build_reference

log = logging.getLogger("escox3.taxonomy")

REQUIRED = ["green_skills.csv", "occupations.csv", "isco_groups.csv", "nace.csv", "occupation_nace.csv",
            "manifest.json", "gaps.json"]
ISCO_LEVEL_NAMES = {"major": "Major group", "sub_major": "Sub-major group", "minor": "Minor group", "unit": "Unit group"}
NACE_LEVEL_NAMES = {"section": "Section", "division": "Division", "group": "Group", "class": "Class"}


class ReferenceStore:
    def __init__(self, reference_dir: Path = REFERENCE_DIR, green_share_threshold: float = 0.10):
        self.dir = Path(reference_dir)
        if not all((self.dir / f).exists() for f in REQUIRED):
            log.info("Reference store incomplete - building it from the supplied raw files ...")
            build_reference.build(self.dir)
        self.green_share_threshold = green_share_threshold
        rd = lambda name: pd.read_csv(self.dir / name, dtype=str, keep_default_na=False)

        self.green_skills = rd("green_skills.csv")
        self.occupations = rd("occupations.csv")
        self.isco = rd("isco_groups.csv")
        self.nace = rd("nace.csv")
        self.occupation_nace = rd("occupation_nace.csv")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.gaps = json.loads((self.dir / "gaps.json").read_text())

        self._skill_idx = {u: i for i, u in enumerate(self.green_skills.uri)}
        self._occ_idx = {u: i for i, u in enumerate(self.occupations.uri)}
        self._isco_idx = {c: i for i, c in enumerate(self.isco.code)}
        self._nace_idx = {c: i for i, c in enumerate(self.nace.code)}
        self.nace_languages = sorted(c.replace("title_", "") for c in self.nace.columns if c.startswith("title_"))

        # optional: official NACE structure (completes parent titles)
        self._nace_extra: Dict[str, str] = {}
        p = self.dir / "nace_structure_imported.csv"
        if p.exists():
            s = pd.read_csv(p, dtype=str, keep_default_na=False)
            self._nace_extra = dict(zip(s.code, s.title_en))

        # optional green occupation references
        self.green_share: Optional[pd.DataFrame] = None
        p = self.dir / "green_occupations.csv"
        if p.exists():
            self.green_share = pd.read_csv(p).set_index("occupation_uri")
        self.green_imported: Optional[pd.DataFrame] = None
        p = self.dir / "green_occupations_imported.csv"
        if p.exists():
            self.green_imported = pd.read_csv(p, dtype=str, keep_default_na=False)

    # ------------------------------------------------------------------ generic
    def resolve_label(self, identifier: str, lang: str = "en") -> Dict[str, object]:
        """Map any supported identifier to its preferred label (or mark it unresolved)."""
        if identifier in self._skill_idx:
            return {"id": identifier, "label": self.green_skills.label.iat[self._skill_idx[identifier]],
                    "type": "green_skill", "resolved": True}
        if identifier in self._occ_idx:
            return {"id": identifier, "label": self.occupations.label.iat[self._occ_idx[identifier]],
                    "type": "esco_occupation", "resolved": True}
        code = identifier.rsplit("/C", 1)[-1] if "/esco/isco/" in identifier else identifier
        if code in self._isco_idx:
            return {"id": identifier, "label": self.isco.label.iat[self._isco_idx[code]],
                    "type": "isco_group", "resolved": True}
        if identifier in self._nace_idx:
            return {"id": identifier, "label": self.nace_title(identifier, lang), "type": "nace", "resolved": True}
        return {"id": identifier, "label": None, "type": "unknown", "resolved": False}

    # ------------------------------------------------------------------ skills
    def skill(self, uri: str) -> Optional[dict]:
        i = self._skill_idx.get(uri)
        if i is None:
            return None
        r = self.green_skills.iloc[i]
        b_uris = [u for u in r.broader_uris.split(" | ") if u]
        b_lbls = r.broader_labels.split(" | ") if r.broader_labels else [""] * len(b_uris)
        return {
            "label": r.label,
            "esco_uri": r.uri,
            "skill_type": r.skill_type,
            "reuse_level": r.reuse_level,
            "description": r.description,
            "alt_labels": [a for a in r.alt_labels.split(" | ") if a],
            "broader": [{"uri": u, "label": (l or None)} for u, l in zip(b_uris, b_lbls)],
            "green": True,
            "green_basis": "Listed in the supplied ESCO green skills collection (greenSkillsCollection_en.csv)",
        }

    # ------------------------------------------------------------------ occupations / ISCO
    def isco_group(self, code: str) -> Optional[dict]:
        i = self._isco_idx.get(code)
        if i is None:
            return None
        r = self.isco.iloc[i]
        return {"code": r.code, "label": r.label, "level": r.level, "level_name": ISCO_LEVEL_NAMES[r.level],
                "isco_uri": r.uri}

    def isco_hierarchy(self, unit_code: str) -> List[dict]:
        if not unit_code:
            return []
        out = []
        for n in (1, 2, 3, 4):
            g = self.isco_group(unit_code[:n])
            out.append(g if g else {"code": unit_code[:n], "label": None, "level": None, "unresolved": True})
        return out

    def occupation(self, uri: str) -> Optional[dict]:
        i = self._occ_idx.get(uri)
        if i is None:
            return None
        r = self.occupations.iloc[i]
        unit = self.isco_group(r.isco_unit) if r.isco_unit else None
        return {
            "label": r.label,
            "esco_uri": r.uri,
            "esco_code": r.esco_code,
            "description": r.description,
            "isco_code": r.isco_unit or None,
            "isco_label": unit["label"] if unit else None,
            "isco_hierarchy": self.isco_hierarchy(r.isco_unit),
            "isco_basis": "ISCO-08 unit group = first 4 digits of the ESCO occupation code in the supplied "
                          "ESCO-NACE crosswalk (ESCO v1.2); validated against the ISCO-08 group list.",
            "nace_codes": [c for c in r.nace_codes.split(" | ") if c],
            "green_status": self.occupation_green_status(r.uri, r.isco_unit),
        }

    def occupation_green_status(self, uri: str, isco_unit: str = "") -> dict:
        if self.green_imported is not None:
            g = self.green_imported
            hit = g[g.occupation_uri == uri] if "occupation_uri" in g.columns else g.iloc[0:0]
            if hit.empty and "isco_code" in g.columns and isco_unit:
                hit = g[g.isco_code == isco_unit]
            if not hit.empty:
                row = hit.iloc[0]
                green = str(row.is_green).lower() in ("true", "1", "yes")
                return {"value": "green" if green else "not_green", "green_share": None,
                        "basis": f"Imported green-occupation list ({row.source})"}
        if self.green_share is not None:
            if uri in self.green_share.index:
                row = self.green_share.loc[uri]
                share = float(row.green_share_all)
                return {"value": "green" if share >= self.green_share_threshold else "not_green",
                        "green_share": round(share, 4),
                        "green_share_essential": float(row.green_share_essential),
                        "n_green_skills": int(row.n_green_all), "n_skills": int(row.n_skills_all),
                        "basis": (f"Green share = {int(row.n_green_all)}/{int(row.n_skills_all)} linked ESCO "
                                  f"skills are in the green skills collection; threshold "
                                  f"{self.green_share_threshold:.2f} (user-configurable, not an official cut-off).")}
            return {"value": "undetermined", "green_share": None,
                    "basis": "Occupation not present in the occupation-skill relation file."}
        return {"value": "undetermined", "green_share": None,
                "basis": "No green-occupation reference was supplied (see data gaps). Status is not guessed."}

    def occupations_for_nace(self, code: str) -> List[str]:
        return self.occupation_nace.occupation_uri[self.occupation_nace.nace_code == code].tolist()

    # ------------------------------------------------------------------ NACE
    def nace_title(self, code: str, lang: str = "en") -> Optional[str]:
        i = self._nace_idx.get(code)
        if i is not None:
            col = f"title_{lang.lower()}"
            t = self.nace[col].iat[i] if col in self.nace.columns else ""
            return t or self.nace["title_en"].iat[i] or None
        return self._nace_extra.get(code)  # imported official structure (English titles only)

    def nace_entry(self, code: str, lang: str = "en") -> Optional[dict]:
        i = self._nace_idx.get(code)
        if i is None:
            return None
        r = self.nace.iloc[i]
        return {
            "label": self.nace_title(code, lang),
            "label_en": r.title_en,
            "nace_code": code,
            "level": r.level,
            "level_name": NACE_LEVEL_NAMES.get(r.level, r.level),
            "hierarchy": self.nace_hierarchy(code, lang),
            "n_linked_occupations": int(r.n_occupations or 0),
            "nace_basis": "NACE Rev. 2.1 code and title from the supplied ESCO-NACE crosswalk",
        }

    def nace_hierarchy(self, code: str, lang: str = "en") -> List[dict]:
        out = []
        for c in build_reference.nace_ancestors(code):
            lvl = build_reference.nace_level(c)
            t = self.nace_title(c, lang)
            out.append({"code": c, "level": lvl, "level_name": NACE_LEVEL_NAMES.get(lvl, lvl), "title": t,
                        "unresolved": t is None})
        return out

    # ------------------------------------------------------------------ info
    def summary(self) -> dict:
        return {"counts": self.manifest.get("counts"), "versions": self.manifest.get("taxonomy_versions"),
                "nace_label_languages": self.nace_languages,
                "green_occupation_reference": ("imported list" if self.green_imported is not None else
                                               "green share (ESCO relations)" if self.green_share is not None
                                               else "missing"),
                "gaps": self.gaps}


@lru_cache(maxsize=4)
def get_store(reference_dir: str = str(REFERENCE_DIR), green_share_threshold: float = 0.10) -> ReferenceStore:
    return ReferenceStore(Path(reference_dir), green_share_threshold)
