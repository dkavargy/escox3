# ESCOX 3.0 evaluation framework

## Experiments

| ID | System | What it measures |
|---|---|---|
| **A** | Original ESCOX algorithm, unchanged (`SkillExtractor._get_entity`: original splitter, best concept per fragment, threshold 0.60 skills / 0.55 occupations; 0.50 for NACE titles) on the ESCOX 3.0 reference sets | semantic retrieval only — the published ESCOX approach |
| **A2** | ESCOX 3.0 retrieval stage only (`mode=retrieval`) | effect of the improved retrieval without LLM |
| **B** | LLM only (`mode=llm_only`): the LLM names concepts without retrieval; mapped by exact label/code lookup | what the LLM achieves alone; hallucination rate |
| **C** | ESCOX 3.0 hybrid (`mode=hybrid`) | full system; C vs A = effect of the LLM component |

Green skills, occupations and sectors are evaluated separately. Occupations are additionally scored at
ISCO-08 unit-group level, sectors at NACE division level.

## Metrics (`metrics.py`)

Micro and macro precision / recall / F1, 95 % bootstrap CI of micro-F1, exact-match accuracy (predicted set
= gold set), invalid-label rate (outputs not in the candidate list / not in the taxonomy, plus unparsable
JSON, divided by raw outputs), mean/median seconds per document, paired bootstrap tests (C vs A, C vs A2,
C vs B, B vs A), breakdowns by `doc_type` and `language`, threshold × top-k sensitivity grid for A2
(`--sweep`) and candidate-threshold × candidate-limit grid for C (`--sweep-hybrid`), and per-document
false positive / false negative lists for error analysis.

## Running

```bash
python evaluation/run_evaluation.py --gold evaluation/gold/<your_gold>.jsonl --experiments A,A2,B,C \
       --sweep --out evaluation/results/run1
```

Outputs: `metrics.csv`, `comparisons.csv`, `sensitivity.csv`, `per_document.csv`, `errors.csv`,
`run_manifest.json` (configuration, LLM name + SHA-256, taxonomy provenance).

## What is needed before results can be reported

**No expert-labelled dataset was supplied, so no experimental results exist yet.** Required:

1. A sample of real documents — suggested ≥ 150, stratified: job ads, CVs, course/training descriptions;
   green and non-green; several EU languages if multilingual robustness is to be claimed.
2. Human gold labels following [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md), in the format of
   [gold/TEMPLATE.jsonl](gold/TEMPLATE.jsonl) / [gold_schema.json](gold_schema.json); ≥ 20 % double-annotated
   with agreement reported.
3. Optional pooling workflow: `build_annotation_pool.py` (blind, shuffled union of system outputs for
   annotators to judge + add missing concepts) → `pool_to_gold.py`.
4. For green-occupation status analyses: the green-occupation reference (see docs/DATA_AND_GAPS.md).

`tests/fixtures/harness_smoke_gold.jsonl` exists only to check that the harness runs; its labels are not
human annotations and must not be reported.
