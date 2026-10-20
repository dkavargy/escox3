<p align="center"><img src="escox3/static/img/logo.png" alt="ESCOX 3.0" width="420"></p>

# ESCOX 3.0 — Green Skills, Occupations & Sector Intelligence

ESCOX 3.0 turns unstructured labour-market text (job advertisements, CVs, course descriptions,
training programmes, professional profiles) into structured, taxonomy-aligned knowledge:

* **Green skills** — concepts of the ESCO green skills collection (570 skills / knowledge concepts)
* **Occupations** — ESCO occupations with **verified ISCO-08 codes** (major → unit group) and green status
* **Economic sectors** — **NACE Rev. 2.1** activities (section → class), titles in 24 EU languages

It extends the open-source ESCOX extractor (kept intact in `esco_skill_extractor/`) with a hybrid
**semantic retrieval + LLM validation** pipeline:

```
Unstructured text → fragments → (1) embedding retrieval of candidates → (2) instruction-tuned LLM reads text + candidates
→ (3) grammar-constrained JSON (IDs restricted to the candidates) → (4) validation against the taxonomy + evidence check
→ green skills · occupations (ISCO-08) · sectors (NACE) → one-page GUI / API / JSON & CSV export
```

Documentation: [Architecture](docs/ARCHITECTURE.md) · [LLM transparency sheet](docs/LLM_COMPONENT.md) ·
[Reference data & gaps](docs/DATA_AND_GAPS.md) · [Evaluation](evaluation/README.md)

---

## 1. Requirements

* Python ≥ 3.10 (tested with 3.13), ~4 GB free RAM (CPU is enough), ~2 GB disk for models
* A C/C++ compiler is needed only if pip has to build `llama-cpp-python` from source
  (Linux: `build-essential cmake`; Windows: Visual Studio Build Tools; macOS: Xcode CLT)
* Internet access on first run (downloads the embedding model and the LLM from Hugging Face)

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: CPU-only PyTorch (smaller):  pip install torch --index-url https://download.pytorch.org/whl/cpu
# optional GPU for the LLM:              CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python
pip install -e .                                          # optional: adds the `escox3` command
```

## 3. Models

| Role | Default | How it is obtained |
|---|---|---|
| Embeddings (retrieval) | `sentence-transformers/all-MiniLM-L6-v2` (same as original ESCOX) | auto-download on first start |
| LLM (contextual validation) | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` → `qwen2.5-1.5b-instruct-q4_k_m.gguf` (Apache-2.0, ~1.1 GB) | auto-download on first start into the Hugging Face cache |

Use a local file or another model:

```bash
python -m escox3 --llm-model-path ~/models/qwen2.5-7b-instruct-q4_k_m.gguf           # any GGUF instruct model
python -m escox3 --llm-repo Qwen/Qwen2.5-7B-Instruct-GGUF --llm-file qwen2.5-7b-instruct-q4_k_m.gguf
python -m escox3 --llm-backend openai --llm-api-base http://localhost:11434/v1 --llm-model qwen2.5:7b-instruct   # Ollama / vLLM / LM Studio
python -m escox3 --llm-backend transformers --llm-model Qwen/Qwen2.5-1.5B-Instruct
python -m escox3 --llm-backend none                                                    # retrieval only
python -m escox3 --embedding-model paraphrase-multilingual-MiniLM-L12-v2               # non-English texts
```

No paid API is needed. An external API key, if you use one, is read from `ESCOX_LLM_API_KEY`.

## 4. Reference data

The supplied files are in `escox3/data/raw/` and are compiled into `escox3/data/reference/`
(already built; rebuild with `python -m escox3 build-reference`). Check counts, versions and **known
gaps** with `python -m escox3 info` or the *Reference data & known gaps* link in the GUI footer.

> **Green occupations:** the supplied files define green *skills* but contain no green-occupation
> classification. Occupation green status is therefore shown as *undetermined* until you add the official
> ESCO `occupationSkillRelations_en.csv` (enables a transparent green-share indicator) or an authoritative
> list — see [docs/DATA_AND_GAPS.md](docs/DATA_AND_GAPS.md).

## 5. Start the application

```bash
python -m escox3                      # = python -m escox3 serve ; then open http://localhost:8000
python -m escox3 --port 8080 --host 0.0.0.0 --config config/escox3.example.toml
```

The first start embeds the reference data (≈5 min on a 2-core CPU, cached afterwards).

**Using the page:** paste text, drop a TXT/PDF/DOCX file or click an example → choose *Green skills*,
*Green occupations*, *Economic sectors* (any combination) → **Analyse Text**. Each result expands to show
the evidence quote, similarity, LLM decision, ISCO/NACE hierarchy and identifiers; the *Evidence in your
text* panel highlights the supporting passages; *Uncertain / rejected candidates* shows what the LLM
rejected and why; **JSON** / **CSV** export everything incl. identifiers. *Advanced settings* switches between
Hybrid, Retrieval-only and LLM-only modes (useful for live comparisons) and the NACE label language.

## 6. API

```bash
curl -X POST http://localhost:8000/api/analyse -H "Content-Type: application/json" -d '{
  "text": "We are looking for an engineer with experience in renewable energy systems.",
  "extract_green_skills": true, "extract_green_occupations": true, "extract_sectors": true,
  "mode": "hybrid", "label_language": "en"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /api/analyse` | analysis → `green_skills`, `green_occupations`, `sectors`, `module_status`, `rejected_candidates`, `metadata` (schema in [ARCHITECTURE.md](docs/ARCHITECTURE.md#4-result-schema-post-apianalyse)) |
| `POST /api/extract-text` | multipart `file` (TXT/PDF/DOCX) → `{text, info}` |
| `POST /api/export/csv`, `/api/export/json` | result → file download |
| `GET /api/health` | models, LLM name/source/SHA-256, decoding parameters |
| `GET /api/taxonomy` | reference counts, versions, data gaps |
| `GET /api/resolve?id=…&lang=…` | identifier → preferred label (or `resolved: false`) |
| `POST /extract-skills`, `POST /extract-occupations` | **original ESCOX endpoints, unchanged** |

Python:

```python
from escox3 import ESCOX3, Config
engine = ESCOX3(Config())
result = engine.analyse(text, extract_green_skills=True, extract_green_occupations=False, extract_sectors=True)
```

Command line: `python -m escox3 analyse --file cv.pdf --modules skills,sectors --csv out.csv`

The original library API still works: `from esco_skill_extractor import SkillExtractor`.

## 7. Example inputs

`evaluation/demo_inputs/` contains job ads (environmental engineer, PV designer, software developer in a
hospital — sector ≠ occupation test, non-green accountant, Greek wind-turbine technician — multilingual
test), a CV and a course description. The GUI has three of them as one-click examples.

## 8. Evaluation

`evaluation/run_evaluation.py` compares **A** original ESCOX matching, **A2** ESCOX 3.0 retrieval only,
**B** LLM only, **C** hybrid ESCOX 3.0 on human-annotated data (precision, recall, F1 with bootstrap CIs,
exact match, invalid/hallucinated-label rate, time, paired significance tests, per document type and
language, threshold and top-k sensitivity, error lists). No annotated dataset was supplied, so **no
results are reported**; see [evaluation/README.md](evaluation/README.md) for what is needed.

## 9. Tests

```bash
pip install -r requirements-dev.txt && python -m pytest -q tests
```

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `llama-cpp-python` fails to build | install a compiler + CMake, or use a prebuilt wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`; or run with `--llm-backend openai` against Ollama |
| Header says "LLM off · retrieval only" | the LLM did not load — see the console log; results are then *not* LLM-validated and the page says so |
| Analysis is slow | expected on CPU (~1–2 min/document on 2 cores); use more threads (`--n-threads`), a GPU build (`--n-gpu-layers -1`) or retrieval mode for quick demos |
| Hugging Face not reachable | download the files elsewhere and use `--embedding-model /path/to/all-MiniLM-L6-v2 --llm-model-path /path/model.gguf` |
| Changed the embedding model | caches are keyed by model; the original ESCOX cache is invalidated automatically |
| PDF gives "no selectable text" | the PDF is a scan — run OCR first |
| Non-English text finds nothing | use a multilingual embedding model (see §3) |
| Port in use | `--port 8080` |

## 11. Credits and licence

MIT licence (see `LICENSE`). Original ESCOX by Konstantinos Petrakis et al.:

> D. C. Kavargyris, K. Georgiou, E. Papaioannou, K. Petrakis, N. Mittas, L. Angelis. *ESCOX: A tool for skill
> and occupation extraction using LLMs from unstructured text.* Software Impacts, 2025, 100772.
> https://doi.org/10.1016/j.simpa.2025.100772

ESCO is © European Union (CC BY 4.0); NACE is published by Eurostat; ISCO-08 by the ILO.
The ESCOX 3.0 logo is an original vector design created for this project.
