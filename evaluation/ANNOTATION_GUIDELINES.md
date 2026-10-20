# ESCOX 3.0 — Annotation guidelines for the gold standard

These guidelines define how human annotators create the reference data used by
`run_evaluation.py`. Model outputs must never be copied into the gold file without a
human judgement; when the pooling workflow is used (`build_annotation_pool.py`), every
system suggestion is accepted or rejected by a person, and the person adds anything the
systems missed.

## Units

One record = one document (job advertisement, CV, course description, training
programme, professional profile …). Record the document type and language.

## 1. Green skills (`gold.green_skills`)

* Only concepts from `escox3/data/reference/green_skills.csv` (the supplied ESCO green skills
  collection) — identified by ESCO URI.
* Annotate a skill if the document **states** it (explicit) or **clearly requires / evidences /
  teaches** it (implicit, e.g. "you will prepare environmental impact studies" →
  *conduct environmental impact assessment*-type concepts).
* Do **not** annotate concepts that are merely topically related (a solar company does not
  imply "assess environmental impact" unless the text says the role does it).
* When two concepts are near-synonyms and both fit, annotate both.

## 2. Occupations (`gold.occupations`)

* ESCO occupation URIs (`occupations.csv`). At most **3** per document, the role the text is about:
  the advertised job, the CV holder's current/target role, or the occupation a course trains for.
* ISCO-08 codes are derived automatically from the ESCO occupation (first four digits of the ESCO
  code) and evaluated as a second, coarser level — do not annotate ISCO separately.
* Green status is *not* annotated: it is a property of the occupation in the green-occupation
  reference, not of the document.

## 3. Economic sectors (`gold.sectors`)

* NACE Rev. 2.1 codes exactly as written in the ESCO–NACE crosswalk (e.g. `D35.11`, `K62`, `R86.1`).
  Use `nace.csv` to look them up. At most **3**.
* Annotate the economic activity of the **employer / organisation / activity described** — not the
  occupation. A software developer employed by a hospital → hospital activities.
* Use the most specific level the text supports; if only a division is supported, use the division.
* If the text gives no evidence about the economic activity, leave the list **empty**.

## Quality control

* Double-annotate at least 20 % of documents and report inter-annotator agreement
  (e.g. Cohen's κ per concept decision in the pooled sheet, or set-based F1 between annotators).
* Resolve disagreements by adjudication and keep the adjudicated file as gold.
* Keep the guidelines version and annotator IDs in the records.
