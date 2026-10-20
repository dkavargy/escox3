"""
Build the unified ESCOX 3.0 reference store from the supplied source files.

Inputs (never modified):
  * escox3/data/raw/greenSkillsCollection_en.csv        – ESCO green skills collection (supplied)
  * escox3/data/raw/ESCO-NACE_rev._2.1_crosswalk.xlsx   – ESCO v1.2 occupation -> NACE Rev. 2.1 crosswalk (supplied)
  * esco_skill_extractor/data/occupations.csv          – original ESCOX: ESCO occupations + ISCO-08 groups
  * esco_skill_extractor/data/skills.csv               – original ESCOX: ESCO skills
  * (optional) escox3/data/raw/occupationSkillRelations_en.csv – official ESCO download, enables green share

Outputs (escox3/data/reference/):
  green_skills.csv, occupations.csv, isco_groups.csv, nace.csv, occupation_nace.csv,
  [green_occupations.csv], manifest.json, gaps.json

Run:  python -m escox3 build-reference
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import RAW_DIR, REFERENCE_DIR, PACKAGE_DIR

log = logging.getLogger("escox3.build")

ORIGINAL_DATA = PACKAGE_DIR.parent / "esco_skill_extractor" / "data"
GREEN_SKILLS_FILE = RAW_DIR / "greenSkillsCollection_en.csv"
CROSSWALK_FILE = RAW_DIR / "ESCO-NACE_rev._2.1_crosswalk.xlsx"
RELATIONS_FILE = RAW_DIR / "occupationSkillRelations_en.csv"

NACE_LANGS = ["DE", "FR", "BG", "CS", "DA", "EL", "ES", "ET", "FI", "GA", "HR", "HU",
              "IT", "LT", "LV", "MT", "NL", "PL", "PT", "RO", "SK", "SL", "SV"]
ISCO_LEVELS = {1: "major", 2: "sub_major", 3: "minor", 4: "unit"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _split_pipe(value) -> List[str]:
    if not isinstance(value, str):
        return []
    return [v.strip() for v in value.split("|") if v.strip()]


def nace_level(code: str) -> Optional[str]:
    if re.fullmatch(r"[A-Z]", code):
        return "section"
    if re.fullmatch(r"[A-Z]\d{2}", code):
        return "division"
    if re.fullmatch(r"[A-Z]\d{2}\.\d", code):
        return "group"
    if re.fullmatch(r"[A-Z]\d{2}\.\d{2}", code):
        return "class"
    return None


def nace_ancestors(code: str) -> List[str]:
    """Return [section, division, group, class] prefixes that exist for a code (incl. itself)."""
    out = [code[0]]
    if len(code) >= 3:
        out.append(code[:3])
    if len(code) >= 5:
        out.append(code[:5])
    if len(code) >= 6:
        out.append(code[:6])
    return out


# --------------------------------------------------------------------------- green skills
def build_green_skills(skills_orig: pd.DataFrame, gaps: List[dict], quality: dict) -> pd.DataFrame:
    g = pd.read_csv(GREEN_SKILLS_FILE, dtype=str, keep_default_na=False)
    expected = ["conceptType", "conceptUri", "preferredLabel", "status", "skillType", "reuseLevel",
                "altLabels", "description", "broaderConceptUri", "broaderConceptPT"]
    missing_cols = [c for c in expected if c not in g.columns]
    if missing_cols:
        raise ValueError(f"Green skills file lacks expected columns: {missing_cols}")

    for c in g.columns:
        g[c] = g[c].str.strip()
    quality["green_skills"] = {
        "rows": int(len(g)),
        "duplicate_uris": int(g.conceptUri.duplicated().sum()),
        "duplicate_preferred_labels": int(g.preferredLabel.duplicated().sum()),
        "empty_altLabels": int((g.altLabels == "").sum()),
        "empty_description": int((g.description == "").sum()),
        "empty_broader": int((g.broaderConceptUri == "").sum()),
        "skillType_counts": g.skillType.value_counts().to_dict(),
        "reuseLevel_counts": g.reuseLevel.value_counts().to_dict(),
        "status_counts": g.status.value_counts().to_dict(),
    }
    g = g.drop_duplicates("conceptUri")

    in_orig = g.conceptUri.isin(set(skills_orig["id"]))
    n_missing = int((~in_orig).sum())
    if n_missing:
        gaps.append({
            "id": "green_skills_not_in_original_escox",
            "severity": "info",
            "detail": f"{n_missing} of {len(g)} green skills are absent from the original ESCOX skills.csv "
                      "(different ESCO release). They are fully supported in ESCOX 3.0 because the green "
                      "module embeds the supplied green-skill file directly.",
            "examples": g.loc[~in_orig, "preferredLabel"].head(10).tolist(),
        })

    rows = []
    for _, r in g.iterrows():
        alts = _split_pipe(r.altLabels)
        b_uris, b_pts = _split_pipe(r.broaderConceptUri), _split_pipe(r.broaderConceptPT)
        if len(b_uris) != len(b_pts):
            # cannot align broader URIs and labels reliably -> keep URIs, mark labels unresolved
            b_pts = [""] * len(b_uris)
        # embedding text follows the original ESCOX recipe: label + alternative labels + description
        embed_text = " ".join([r.preferredLabel] + alts + [r.description]).strip()
        rows.append({
            "uri": r.conceptUri,
            "label": r.preferredLabel,
            "alt_labels": " | ".join(alts),
            "description": r.description,
            "concept_type": r.conceptType,
            "skill_type": r.skillType,
            "reuse_level": r.reuseLevel,
            "status": r.status,
            "broader_uris": " | ".join(b_uris),
            "broader_labels": " | ".join(b_pts),
            "in_original_escox": bool(r.conceptUri in set(skills_orig["id"])),
            "embed_text": embed_text,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ISCO / occupations / NACE
def build_isco(occ_orig: pd.DataFrame, quality: dict) -> pd.DataFrame:
    isco = occ_orig[occ_orig["id"].str.contains("/esco/isco/")].copy()
    isco["code"] = isco["id"].str.extract(r"/isco/C(\d+)$")[0]
    # In the original ESCOX file the ISCO description starts with the group title followed by two spaces.
    isco["label"] = isco["description"].str.split("  ", n=1, regex=False).str[0].str.strip()
    isco["level"] = isco["code"].str.len().map(ISCO_LEVELS)
    isco["description_text"] = isco["description"].str.split("  ", n=1, regex=False).str[1].fillna("").str.strip()
    out = isco[["code", "level", "label", "id", "description_text"]].rename(columns={"id": "uri"})
    quality["isco_groups"] = {"rows": int(len(out)), "level_counts": out.level.value_counts().to_dict()}
    return out.sort_values("code").reset_index(drop=True)


def build_occupations_and_nace(occ_orig: pd.DataFrame, isco: pd.DataFrame, gaps: List[dict], quality: dict):
    xw = pd.read_excel(CROSSWALK_FILE, sheet_name="ESCO-NACE crosswalk", dtype=str).fillna("")
    versions = pd.read_excel(CROSSWALK_FILE, sheet_name="ESCO and NACE versions used", header=None, dtype=str)
    version_info = {}
    for _, row in versions.iterrows():
        k, v = row.iloc[0], row.iloc[1] if len(row) > 1 else None
        if isinstance(k, str) and isinstance(v, str):
            version_info[k.strip().rstrip(":")] = v.strip()

    for c in ["ESCO URI", "ESCO Code", "ESCO PT", "NACE code", "NACE title"]:
        xw[c] = xw[c].str.strip()

    quality["crosswalk"] = {
        "rows": int(len(xw)),
        "unique_occupations": int(xw["ESCO URI"].nunique()),
        "unique_nace_codes": int(xw["NACE code"].nunique()),
        "duplicate_rows_uri_nace": int(xw.duplicated(["ESCO URI", "NACE code"]).sum()),
        "empty_cells": {c: int((xw[c] == "").sum()) for c in xw.columns if (xw[c] == "").any()},
        "nace_codes_per_occupation": xw.groupby("ESCO URI").size().describe().round(3).to_dict(),
        "versions": version_info,
    }
    xw = xw.drop_duplicates(["ESCO URI", "NACE code"])

    # ---- NACE reference
    bad = xw[xw["NACE code"].map(nace_level).isna()]
    if len(bad):
        gaps.append({"id": "invalid_nace_codes", "severity": "warning",
                     "detail": f"{len(bad)} crosswalk rows have NACE codes with an unrecognised format; ignored.",
                     "examples": bad["NACE code"].head(10).tolist()})
        xw = xw[xw["NACE code"].map(nace_level).notna()]

    title_conflicts = xw.groupby("NACE code")["NACE title"].nunique()
    if (title_conflicts > 1).any():
        gaps.append({"id": "nace_title_conflicts", "severity": "warning",
                     "detail": "Some NACE codes have more than one English title in the crosswalk; first kept.",
                     "examples": title_conflicts[title_conflicts > 1].index[:10].tolist()})

    lang_cols = {"EN": "NACE title", **{l: f"{l} title" for l in NACE_LANGS if f"{l} title" in xw.columns}}
    nace = xw.drop_duplicates("NACE code")[["NACE code"] + list(lang_cols.values())].copy()
    nace = nace.rename(columns={"NACE code": "code", **{v: f"title_{k.lower()}" for k, v in lang_cols.items()}})
    nace["level"] = nace["code"].map(nace_level)
    nace["section"] = nace["code"].str[0]
    nace["n_occupations"] = nace["code"].map(xw.groupby("NACE code")["ESCO URI"].nunique())
    nace = nace.sort_values("code").reset_index(drop=True)

    known = set(nace.code)
    missing_parents = sorted({a for c in known for a in nace_ancestors(c)[:-1] if a not in known})
    sections_missing = [m for m in missing_parents if len(m) == 1]
    gaps.append({
        "id": "nace_hierarchy_titles_incomplete",
        "severity": "info",
        "detail": (f"The crosswalk lists {len(known)} NACE codes at mixed levels "
                   f"({nace.level.value_counts().to_dict()}). Titles of {len(missing_parents)} parent codes "
                   f"(including {len(sections_missing)} sections) are not present in the supplied file. ESCOX 3.0 "
                   "shows those parent codes without a title rather than inventing one. Import the official "
                   "Eurostat NACE Rev. 2.1 structure to complete the hierarchy (python -m escox3 import-nace <csv>)."),
        "examples": missing_parents[:15],
    })

    # ---- occupations
    occ_only = occ_orig[occ_orig["id"].str.contains("/esco/occupation/")]
    occ_text = dict(zip(occ_only["id"], occ_only["description"]))
    occ = xw.drop_duplicates("ESCO URI")[["ESCO URI", "ESCO Code", "ESCO PT", "ESCO description"]].copy()
    occ.columns = ["uri", "esco_code", "label", "description"]
    occ["isco_unit"] = occ["esco_code"].str.extract(r"^(\d{4})")[0]
    isco_codes = set(isco.code)
    bad_isco = occ[~occ.isco_unit.isin(isco_codes)]
    if len(bad_isco):
        gaps.append({"id": "isco_unit_not_found", "severity": "warning",
                     "detail": f"{len(bad_isco)} occupations have an ESCO code whose 4-digit prefix is not an "
                               "ISCO-08 unit group in the reference; their ISCO code is left empty.",
                     "examples": bad_isco.label.head(10).tolist()})
        occ.loc[~occ.isco_unit.isin(isco_codes), "isco_unit"] = ""
    occ["isco_minor"] = occ.isco_unit.str[:3]
    occ["isco_sub_major"] = occ.isco_unit.str[:2]
    occ["isco_major"] = occ.isco_unit.str[:1]
    occ["nace_codes"] = occ.uri.map(xw.groupby("ESCO URI")["NACE code"].apply(lambda s: " | ".join(sorted(s))))
    occ["in_original_escox"] = occ.uri.isin(occ_text.keys())
    # Embedding text: identical to the text embedded by the original ESCOX (label + alt labels + description),
    # so that the original cached embeddings can be reused verbatim.
    occ["embed_text"] = occ.uri.map(occ_text).fillna(occ.label + " " + occ.description)
    missing_in_orig = int((~occ.in_original_escox).sum())
    orig_not_in_xw = sorted(set(occ_text) - set(occ.uri))
    quality["occupations"] = {"rows": int(len(occ)), "not_in_original_escox": missing_in_orig,
                              "original_occupations_without_crosswalk": len(orig_not_in_xw)}
    if orig_not_in_xw:
        gaps.append({"id": "occupations_without_nace", "severity": "info",
                     "detail": f"{len(orig_not_in_xw)} original ESCOX occupations have no crosswalk entry."})

    occ_nace = xw[["ESCO URI", "NACE code"]].rename(columns={"ESCO URI": "occupation_uri", "NACE code": "nace_code"})
    occ = occ.sort_values("label").reset_index(drop=True)
    return occ, nace, occ_nace, version_info


# --------------------------------------------------------------------------- green occupations (optional)
def build_green_occupations(green: pd.DataFrame, occ: pd.DataFrame, gaps: List[dict], quality: dict,
                            relations_file: Path = RELATIONS_FILE) -> Optional[pd.DataFrame]:
    """Compute the green share of each occupation from the official ESCO occupation-skill relations.

    green_share = |green skills/knowledge linked to the occupation| / |all skills/knowledge linked|,
    computed separately for essential relations and for essential+optional relations.
    Requires the official ESCO file ``occupationSkillRelations_en.csv`` (not supplied).
    """
    if not relations_file.exists():
        gaps.append({
            "id": "green_occupation_reference_missing",
            "severity": "critical",
            "detail": ("No green-occupation definition or occupation-skill relation file was supplied. The "
                       "supplied files contain green SKILLS (greenSkillsCollection_en.csv) and an occupation->NACE "
                       "crosswalk, but nothing that labels ESCO occupations or ISCO groups as green. ESCOX 3.0 "
                       "therefore identifies occupations and their verified ISCO-08 codes, but reports their green "
                       "status as 'undetermined' instead of guessing. To enable it, place the official ESCO "
                       "download occupationSkillRelations_en.csv (same ESCO release as the crosswalk, v1.2) in "
                       "escox3/data/raw/ and run `python -m escox3 build-reference`, or import an authoritative "
                       "green-occupation list with `python -m escox3 import-green-occupations <csv>`."),
        })
        return None

    rel = pd.read_csv(relations_file, dtype=str, keep_default_na=False)
    need = {"occupationUri", "relationType", "skillUri"}
    if not need.issubset(rel.columns):
        raise ValueError(f"{relations_file.name} must contain columns {need}; found {list(rel.columns)}")
    rel["green"] = rel.skillUri.isin(set(green.uri))
    out = []
    for uri, grp in rel.groupby("occupationUri"):
        ess = grp[grp.relationType == "essential"]
        out.append({
            "occupation_uri": uri,
            "n_skills_all": len(grp), "n_green_all": int(grp.green.sum()),
            "green_share_all": round(grp.green.mean(), 4) if len(grp) else 0.0,
            "n_skills_essential": len(ess), "n_green_essential": int(ess.green.sum()),
            "green_share_essential": round(ess.green.mean(), 4) if len(ess) else 0.0,
            "source": f"derived: {relations_file.name} (sha256 {sha256(relations_file)[:12]}) x greenSkillsCollection_en.csv",
        })
    df = pd.DataFrame(out)
    quality["green_occupations"] = {"rows": int(len(df)), "occupations_with_any_green_skill": int((df.n_green_all > 0).sum())}
    return df


# --------------------------------------------------------------------------- main
def build(reference_dir: Path = REFERENCE_DIR) -> dict:
    reference_dir.mkdir(parents=True, exist_ok=True)
    gaps: List[dict] = []
    quality: Dict[str, dict] = {}

    skills_orig = pd.read_csv(ORIGINAL_DATA / "skills.csv", dtype=str)
    occ_orig = pd.read_csv(ORIGINAL_DATA / "occupations.csv", dtype=str)

    green = build_green_skills(skills_orig, gaps, quality)
    isco = build_isco(occ_orig, quality)
    occ, nace, occ_nace, versions = build_occupations_and_nace(occ_orig, isco, gaps, quality)
    green_occ = build_green_occupations(green, occ, gaps, quality)

    # carry over an imported green-occupation list if one exists
    imported = reference_dir / "green_occupations_imported.csv"
    if imported.exists():
        gaps[:] = [g for g in gaps if g["id"] != "green_occupation_reference_missing"]

    green.to_csv(reference_dir / "green_skills.csv", index=False)
    isco.to_csv(reference_dir / "isco_groups.csv", index=False)
    occ.to_csv(reference_dir / "occupations.csv", index=False)
    nace.to_csv(reference_dir / "nace.csv", index=False)
    occ_nace.to_csv(reference_dir / "occupation_nace.csv", index=False)
    gs = reference_dir / "green_occupations.csv"
    if green_occ is not None:
        green_occ.to_csv(gs, index=False)
    elif gs.exists():
        gs.unlink()

    gaps.append({
        "id": "green_skills_version_unstated",
        "severity": "info",
        "detail": "greenSkillsCollection_en.csv does not state its ESCO release; the crosswalk states ESCO "
                  f"{versions.get('ESCO version', '?')}. Record the release used in publications.",
    })

    sources = []
    for p in [GREEN_SKILLS_FILE, CROSSWALK_FILE, ORIGINAL_DATA / "skills.csv", ORIGINAL_DATA / "occupations.csv"] + (
            [RELATIONS_FILE] if RELATIONS_FILE.exists() else []):
        sources.append({"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size})
    manifest = {
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "taxonomy_versions": {
            "ESCO (crosswalk)": versions.get("ESCO version"),
            "NACE": "Rev. " + str(versions.get("NACE revision")),
            "ISCO": "ISCO-08 (groups as distributed with the original ESCOX occupations.csv)",
            "green skills": "ESCO green skills collection (release not stated in file)",
        },
        "counts": {
            "green_skills": int(len(green)), "occupations": int(len(occ)), "isco_groups": int(len(isco)),
            "nace_codes": int(len(nace)), "occupation_nace_links": int(len(occ_nace)),
            "green_occupation_shares": int(len(green_occ)) if green_occ is not None else 0,
        },
        "data_quality": quality,
    }
    (reference_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (reference_dir / "gaps.json").write_text(json.dumps(gaps, indent=2, ensure_ascii=False))
    log.info("Reference store built in %s: %s", reference_dir, manifest["counts"])
    return manifest


def import_green_occupations(csv_path: str, reference_dir: Path = REFERENCE_DIR) -> int:
    """Import an authoritative list of green occupations.

    The CSV must contain a column ``occupation_uri`` (ESCO occupation URI) and/or ``isco_code``
    (ISCO-08 unit group), a boolean-like column ``is_green`` and a free-text ``source`` column.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "is_green" not in df.columns or not ({"occupation_uri", "isco_code"} & set(df.columns)):
        raise ValueError("CSV needs 'is_green' plus 'occupation_uri' and/or 'isco_code' columns.")
    if "source" not in df.columns:
        raise ValueError("CSV needs a 'source' column documenting where the green labels come from.")
    df["is_green"] = df["is_green"].str.lower().isin(["1", "true", "yes", "y", "green"])
    df.to_csv(reference_dir / "green_occupations_imported.csv", index=False)
    return len(df)


def import_nace_structure(csv_path: str, reference_dir: Path = REFERENCE_DIR) -> int:
    """Import an official NACE Rev. 2.1 structure CSV (columns: code, title) to complete hierarchy titles.

    Codes may be given as in Eurostat (e.g. 'A', '01', '01.1', '01.11') — they are converted to the
    crosswalk notation (section letter prefix, e.g. 'A01.11') using the section of each division.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    cols = {c.lower(): c for c in df.columns}
    if "code" not in cols or "title" not in cols:
        raise ValueError("NACE structure CSV must have 'code' and 'title' columns.")
    df = df.rename(columns={cols["code"]: "code", cols["title"]: "title"})[["code", "title"]]
    section = None
    rows = []
    for code, title in zip(df.code.str.strip(), df.title.str.strip()):
        if re.fullmatch(r"[A-Z]", code):
            section = code
            rows.append((code, title))
        elif re.fullmatch(r"[A-Z]\d{2}(\.\d{1,2})?", code):
            rows.append((code, title))
        elif section and re.fullmatch(r"\d{2}(\.\d{1,2})?", code):
            rows.append((section + code, title))
    out = pd.DataFrame(rows, columns=["code", "title_en"])
    out.to_csv(reference_dir / "nace_structure_imported.csv", index=False)
    return len(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build()["counts"], indent=2))
