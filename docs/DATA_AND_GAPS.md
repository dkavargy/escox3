# Reference data: inspection report, unified representation and gaps

All counts below were computed by `escox3/build_reference.py` from the supplied files
(`python -m escox3 info` prints the live version, including `gaps.json`).

## 1. Supplied files

### `greenSkillsCollection_en.csv` (SHA-256 `a9494b98…7f21`)

| Item | Finding |
|---|---|
| Format | CSV, UTF-8, comma-separated, 570 rows, 10 columns |
| Columns | `conceptType, conceptUri, preferredLabel, status, skillType, reuseLevel, altLabels, description, broaderConceptUri, broaderConceptPT` |
| Identifier | `conceptUri` = ESCO skill URI (570 unique) |
| Names | `preferredLabel` (570 unique, English); `altLabels` pipe-separated (38 rows empty) |
| Types | skillType: 384 skill/competence, 186 knowledge; reuseLevel: 329 sector-specific, 166 cross-sector, 70 occupation-specific, 5 transversal; status all `released`; conceptType all `KnowledgeSkillCompetence` |
| Hierarchy | `broaderConceptUri`/`broaderConceptPT` pipe-separated and aligned (used for "broader concept") |
| Occupations / ISCO / NACE | **none** — the file is a skill collection only |
| Quality | no duplicate URIs or labels; empty cells are blank strings, not NaN; release not stated |
| vs. original ESCOX | 564 of 570 URIs exist in the original `skills.csv`; 6 do not (newer ESCO release) — supported because the green index is built from this file |

### `ESCO-NACE_rev._2.1_crosswalk.xlsx` (SHA-256 `b0e6db85…f9e`)

| Item | Finding |
|---|---|
| Sheets | `ESCO and NACE versions used` (ESCO 1.2, NACE Rev. 2.1); `ESCO-NACE crosswalk` (4 564 rows × 29 columns) |
| Columns | `ESCO URI, ESCO Code, ESCO PT, ESCO description, NACE code, NACE title` + NACE titles in 23 further EU languages (DE … SV) |
| Occupations | 3 039 unique ESCO occupations = exactly the occupations of the original ESCOX file |
| ISCO-08 | not a separate column; `ESCO Code` (e.g. `2166.3.1`) starts with the 4-digit ISCO-08 unit group — every one validated against the ISCO-08 group list |
| NACE | 706 unique codes at mixed levels: 9 sections, 54 divisions, 219 groups, 424 classes; notation = section letter + code (`D35.12`) |
| Relationships | occupation → 1–25 NACE codes (mean 1.5) |
| Quality | no empty cells, no duplicate (occupation, NACE) pairs, one English title per code |

### Original ESCOX data (reused)

`esco_skill_extractor/data/occupations.csv` contains 3 039 ESCO occupations **and 619 ISCO-08 groups**
(10 major, 43 sub-major, 130 minor, 436 unit groups) — the source of ISCO labels and hierarchy.
`skills.csv` (13 939 ESCO skills) remains used by the original endpoints.

## 2. Unified internal representation (`escox3/data/reference/`)

| File | Key | Content |
|---|---|---|
| `green_skills.csv` | ESCO skill URI | label, alt labels, description, type, reuse level, broader concepts, embedding text |
| `occupations.csv` | ESCO occupation URI | label (ESCO PT), ESCO code, ISCO unit/minor/sub-major/major codes, linked NACE codes, embedding text |
| `isco_groups.csv` | ISCO-08 code | level, label, ESCO ISCO URI |
| `nace.csv` | NACE code | level, section, titles in 24 languages, number of linked occupations |
| `occupation_nace.csv` | (occupation URI, NACE code) | crosswalk links |
| `green_occupations.csv` | occupation URI | *only if the ESCO relations file is provided* — green share |
| `manifest.json`, `gaps.json` | — | provenance (SHA-256 of sources), versions, data-quality statistics, gaps |

ESCO URIs, ISCO codes and NACE codes are kept as separate fields; an ISCO code is never presented as an
ESCO URI. `taxonomy.ReferenceStore.resolve_label()` maps any identifier to its exact label or returns
`resolved: false` — labels are never invented.

## 3. Gaps (not silently filled)

1. **No green-occupation definition was supplied (critical).** Nothing in the files labels occupations as
   green. ESCOX 3.0 therefore identifies occupations with verified ISCO-08 codes but reports
   *green status: undetermined*. Two ways to close the gap:
   * put the official ESCO v1.2 download `occupationSkillRelations_en.csv` into `escox3/data/raw/` and run
     `python -m escox3 build-reference` → each occupation gets a **green share** (green skills / all linked
     skills, essential and essential+optional). Status = green if share ≥ `green_share_threshold`
     (default 0.10). *This threshold is a researcher-set parameter, not an official definition* — report it.
   * or import an authoritative list: `python -m escox3 import-green-occupations list.csv`
     (columns `occupation_uri` and/or `isco_code`, `is_green`, `source`).
2. **NACE hierarchy titles incomplete (info).** 103 parent codes (including 13 sections) occurring in the
   hierarchy of the crosswalk codes have no title in the file; they are shown as code + "title not in supplied
   data". Import the Eurostat NACE Rev. 2.1 structure with `python -m escox3 import-nace structure.csv`
   (columns `code,title`) to complete them.
3. **Green skills release not stated** in the CSV (crosswalk is ESCO v1.2). Record it in publications.
4. **Labels are English only** for skills and occupations (only English files supplied); NACE titles are
   multilingual. Multilingual ESCO labels can be added by supplying the corresponding ESCO language files.
5. **No green flag for sectors** exists in the supplied data; sectors are classified, not labelled green.

The official ESCO and Eurostat portals were not reachable from the build environment, so no external file
was downloaded; the importers above accept them locally.
