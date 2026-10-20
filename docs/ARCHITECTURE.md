# Architecture: from ESCOX to ESCOX 3.0

## 1. The original ESCOX (as supplied, `esco-skill-extractor` v0.1.18)

| Aspect | Original implementation |
|---|---|
| Package | `esco_skill_extractor/` — `__init__.py` (class `SkillExtractor`), `__main__.py` (Flask + waitress server) |
| Entry point | `python -m esco_skill_extractor` (args: model, thresholds, device, host, port) |
| Dependencies | sentence-transformers, Flask, pandas, waitress |
| Data | `data/skills.csv` (13 939 ESCO skills: `id, description` where description = label + alt labels + definition); `data/occupations.csv` (3 039 ESCO occupations + 619 ISCO-08 groups, same format) |
| Model | `all-MiniLM-L6-v2` sentence-transformer |
| Mechanism | split text with `re.split(r"\r|\n|\t|\.|\,|\;|and|or")`, embed every fragment, cosine similarity with all reference embeddings, keep the single best concept per fragment if its score > threshold (skills 0.6, occupations 0.55) |
| Caching | reference embeddings pickled into the package `data/` folder (not keyed by model name) |
| API | `POST /extract-skills`, `POST /extract-occupations` (list of texts → list of URI lists) |
| GUI | single page, radio choice skills/occupations, list of raw URIs, "copy CSV" |
| LLM | **none in the code** — only an embedding encoder |
| Database | none |

Observed limitations: no generative LLM; raw URIs in the GUI; one concept per fragment and no scores
or evidence returned; the `and|or` split also cuts inside words ("inf|ormation"); ISCO groups and ESCO
occupations mixed in one result list; cache not invalidated when the model changes.

## 2. What ESCOX 3.0 reuses unchanged

* The **entire `esco_skill_extractor` package** is kept byte-for-byte (including its endpoints, which the
  ESCOX 3.0 server still exposes at the same paths). Verified: `POST /extract-skills` returns the same IDs
  as the example in the original README.
* The **embedding model and the embedding recipe** (label + alt labels + description).
* The **original occupation embeddings**: ESCOX 3.0 instantiates `SkillExtractor` and slices its cached
  occupation matrix instead of re-embedding.
* The **original matching algorithm** is called as-is (`SkillExtractor._get_entity`) for Experiment A.
* Flask + waitress serving, CLI style.

## 3. Minimal architectural additions (new package `escox3/`)

```
escox3/
  config.py            one dataclass for every parameter (TOML / env overrides) -> reproducible runs
  build_reference.py   supplied files -> unified reference store + provenance manifest + gaps report
  taxonomy.py          ReferenceStore: lookup, ISCO / NACE hierarchy, green status, id -> label mapping layer
  retrieval.py         Stage 1: offset-aware fragmenting, cached indices, top-k candidate retrieval
  llm/backends.py      Stage 2: llama.cpp (default) | transformers | OpenAI-compatible | none
  llm/prompts.py       versioned prompt templates + JSON schemas (Stage 3 constraints)
  validation.py        Stage 4: JSON parsing, candidate/structure checks, evidence verification
  pipeline.py          ESCOX3 class: modules, modes (hybrid / retrieval / llm_only), result schema
  text_extraction.py   TXT / PDF / DOCX -> text
  export.py            CSV / JSON export
  server.py            Flask app: GUI + /api/* + original endpoints
  templates/, static/  one-page GUI, logo, favicon
evaluation/            Experiments A, A2, B, C, metrics, sensitivity analysis, annotation tooling
tests/                 unit tests (no model needed)
```

Design decisions:

* **Separate module per output category** with identical structure (retrieve → LLM select → validate →
  build item). Disabled modules are not computed; occupation retrieval is computed for sectors only when the
  crosswalk option is on (cheap vector search, no LLM call).
* **Occupation ≠ sector.** Sector candidates come from semantic similarity of the text with NACE titles;
  crosswalk codes of the identified occupations are added only as extra candidates, and the LLM prompt
  explicitly asks for the employer's economic activity. No evidence → "Sector not confidently identified."
* **IDs vs labels.** Items carry `esco_uri`, `esco_code`, `isco_code`, `nace_code` separately; the GUI
  shows labels and keeps identifiers in expandable details and exports.
* **Determinism.** Greedy decoding, fixed seed, KV-cache reset per call, cached embeddings keyed by
  model + content hash, SHA-256 of sources and model file in metadata.
* **No heavy infrastructure.** No database; CSV reference store (~6 MB) loaded in memory; one model
  instance shared by all requests (thread lock around the LLM).

## 4. Result schema (`POST /api/analyse`)

```json
{
  "green_skills":      [ {"rank", "label", "esco_uri", "skill_type", "reuse_level", "description", "broader", "green", "green_basis",
                          "support": "explicit|implicit|semantic_similarity", "evidence", "evidence_verified", "evidence_span",
                          "similarity", "retrieval_rank", "matched_fragment", "candidate_source", "llm_decision", "confidence", "method"} ],
  "green_occupations": [ {"rank", "label", "esco_uri", "esco_code", "description", "isco_code", "isco_label",
                          "isco_hierarchy": [{"code","label","level"} x4], "isco_basis", "nace_codes",
                          "green_status": {"value": "green|not_green|undetermined", "green_share", "basis"}, ...same evidence fields} ],
  "sectors":           [ {"rank", "label", "label_en", "nace_code", "level", "level_name",
                          "hierarchy": [{"code","level","title","unresolved"}], "nace_basis", ...same evidence fields} ],
  "module_status":     {"green_skills": {"status": "ok|none_found|not_requested", "message", "n_results"}, ...},
  "rejected_candidates": {"<module>": [{"id", "label", "similarity", "reason"}]},
  "metadata": {"escox_version", "model", "processing_time", "mode", "modules_requested", "llm_invoked",
               "llm_calls": [{"module","invoked","n_candidates","latency_s","prompt_tokens","completion_tokens",
                              "grammar_constrained","valid_json","validation","raw_output"}],
               "prompt_version", "models", "timings", "text": {"chars","sha256"}, "taxonomy",
               "green_occupation_reference", "retrieval_config", "warnings"}
}
```
