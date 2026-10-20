# The LLM component of ESCOX 3.0 — transparency sheet

This document is written so that a manuscript can describe the LLM contribution accurately.

## 1. What the LLM does (and does not do)

ESCOX 3.0 uses a **pretrained, instruction-tuned, open-weight LLM for inference only**. It is **not
trained or fine-tuned by ESCOX** and nothing in this repository supports a claim that it was trained on
job advertisements or CVs. Its role is contextual reasoning over a short list of taxonomy candidates:

| Stage | Component | Genuine LLM? |
|---|---|---|
| 1 Candidate retrieval | sentence-transformer embeddings (`all-MiniLM-L6-v2`, as in the original ESCOX) + cosine similarity, top-k per text fragment | No (encoder model, similarity search) |
| 2 Contextual reasoning | instruction-following LLM reads the full text and the numbered candidates and decides which are supported, whether explicitly or implicitly, and quotes evidence | **Yes — autoregressive generation** |
| 3 Taxonomy-constrained output | JSON schema → GBNF grammar in llama.cpp; candidate IDs are an `enum`, so the decoder cannot emit an identifier outside the retrieved list | Constrains the LLM |
| 4 Validation | parse JSON, reject unknown IDs / duplicates / invalid fields / over-long lists, verify each evidence quote against the source text (exact or ≥ 85/100 fuzzy match), map IDs back to taxonomy entries | No (deterministic code) |

Every analysis records whether the LLM was invoked, per-module latency, prompt/completion tokens,
whether grammar-constrained decoding was active, the validation outcome and the raw model output
(`metadata.llm_calls`). The GUI shows the same under *Technical details*.

The original ESCOX paper described the tool as LLM-based although the released code only used
sentence embeddings + cosine similarity. ESCOX 3.0 keeps that mechanism as Stage 1 and adds the
generative LLM in Stage 2; the evaluation framework (Experiments A vs C) measures its effect.

## 2. Default model and justification

| Property | Value |
|---|---|
| Model | **Qwen2.5-1.5B-Instruct** (Alibaba Qwen team) |
| File | `qwen2.5-1.5b-instruct-q4_k_m.gguf` from the official repo `Qwen/Qwen2.5-1.5B-Instruct-GGUF` on Hugging Face |
| Quantisation | Q4_K_M (≈1.1 GB) |
| Licence | Apache-2.0 (check the model card for the version you download) |
| Usage | inference only, no fine-tuning |
| Runtime | llama-cpp-python (llama.cpp), CPU by default, optional GPU offload |
| Context | 8 192 tokens configured (model supports 32k) |
| Languages | multilingual (Qwen2.5 reports support for 29+ languages incl. most EU languages) |

Selection criteria: (1) permissive open licence so the tool can stay open-source and free to run;
(2) runs on an ordinary laptop CPU with ~4 GB free RAM, which matters for live demonstrations;
(3) multilingual coverage for European documents; (4) good instruction following and JSON output for
its size; (5) mature GGUF support in llama.cpp, enabling grammar-constrained decoding.

Alternatives supported by configuration (no code change):

| Model | Why you might choose it | Caveat |
|---|---|---|
| Qwen2.5-7B-Instruct (GGUF Q4_K_M, ≈4.7 GB) | clearly stronger reasoning — recommended for the paper's main experiments if a GPU or ≥ 8 GB RAM is available | slower on CPU |
| Qwen2.5-3B-Instruct | middle ground | Qwen research licence (non-commercial) |
| Llama-3.2-3B-Instruct | strong English instruction following | Llama community licence; fewer official languages |
| Phi-3.5-mini-instruct | MIT licence | weaker multilingual coverage |
| Any model behind Ollama / vLLM / LM Studio / OpenAI API | `--llm-backend openai --llm-api-base …` | an external API is optional, never required |

Report the exact model file and its SHA-256 (shown in `/api/health` and in every result's metadata).

## 3. Decoding parameters (defaults)

`temperature = 0.0` (greedy), `top_p = 1.0`, `seed = 42`, KV-cache reset before each call,
`max_new_tokens = min(700, 60 + 70 × max_items)`, `max_items` = 15 (skills), 3 (occupations), 3 (sectors),
evidence ≤ 200 characters, grammar-constrained JSON on. Inputs longer than 6 000 characters are
condensed to the first 3 300 characters plus the sentences in which candidates were retrieved;
evidence is still verified against the full text. If the prompt exceeds the context window, candidate
descriptions are shortened, then the lowest-ranked candidates are dropped (recorded in
`prompt_adjustments`).

## 4. Candidate retrieval strategy

* Text → sentence and clause fragments (word-boundary split on `and/or/,/;/…`) with character offsets,
  plus one document-level fragment.
* Reference texts are embedded once (label + alternative labels + description, the original ESCOX recipe)
  and cached; occupation embeddings are reused directly from the original ESCOX cache.
* For each fragment the top-k (5) concepts are taken; scores are max-pooled per concept; concepts above
  the candidate threshold (skills 0.40, occupations 0.35, sectors 0.30) are ranked; at most 20 / 15 / 15
  go to the LLM.
* Sectors additionally receive, as **candidates only**, the NACE codes linked by the ESCO–NACE crosswalk
  to the best occupations; the LLM must still find evidence of the economic activity in the text.

## 5. Prompt templates (version `escox3-prompts-v1.0`)

System message (all modules):

```
You are an expert annotator of European labour-market documents. You classify text using the ESCO skills
and occupations taxonomy, the ISCO-08 occupation classification and the NACE Rev. 2.1 classification of
economic activities. You are strict: you only choose concepts that the text genuinely supports, you never
invent concepts, and you reply with a single JSON object only.
```

User message (sector module shown; the TASK line differs per module, see `escox3/llm/prompts.py`):

```
TASK: Select the NACE Rev. 2.1 ECONOMIC ACTIVITIES that describe the business activity of the employer or
organisation, or the economic activity the TEXT is about. A sector is what the organisation produces or
does, NOT the job title: e.g. a software developer working for a hospital belongs to hospital activities,
not to computer programming. Choose at most 3, best match first. If the text gives no evidence about the
economic activity, return an empty list.

Rules:
- You may ONLY use ids from the CANDIDATES list.
- "explicit" = the concept is directly stated in the text (same or equivalent wording).
- "implicit" = the concept is not named but is clearly required, possessed, practised or taught according to the text.
- Do NOT select a candidate that is only loosely related to the topic of the text.
- "evidence" must be a short quotation copied exactly from the TEXT (max 25 words).
- If no candidate is supported, return {"selected": []}.

TEXT:
"""
<document>
"""

CANDIDATES (id | NACE code | economic activity):
N1 | D35.12 | Production of electricity from renewable sources
...

Reply with JSON only, in this format:
{"selected": [{"id": "<candidate id>", "support": "explicit" or "implicit", "evidence": "<exact quote from the TEXT>"}]}
```

JSON schema used for constrained decoding (IDs are the candidates of that call):

```json
{"type":"object","properties":{"selected":{"type":"array","maxItems":3,"items":{"type":"object",
 "properties":{"id":{"type":"string","enum":["N1","N2"]},"support":{"type":"string","enum":["explicit","implicit"]},
 "evidence":{"type":"string","maxLength":200}},"required":["id","support","evidence"]}}},"required":["selected"]}
```

Experiment B (LLM only) uses `build_open_messages` / `open_schema` in the same file: the model names
labels/codes freely and they are matched to the taxonomy by exact (normalised) label or code.

## 6. Validation rules and result semantics

* `explicit` / `inferred (implicit)` — LLM judgement, kept only if the evidence quote is found in the text.
* Rejected / uncertain candidates are returned in `rejected_candidates` with a reason: *rejected by LLM*,
  *unsupported (quote not found)*, *id_not_in_candidates*, *duplicate*, *exceeds maximum*.
* Confidence: **high** = explicit + verified quote; **medium** = implicit + verified quote
  (retrieval-only mode: high ≥ threshold + 0.15, medium ≥ threshold).
* If the LLM cannot be loaded, the system says so in `metadata.warnings` and falls back to retrieval only;
  it never presents unvalidated results as LLM-validated.

## 7. Hardware

| Setting | Requirement (Qwen2.5-1.5B Q4_K_M) |
|---|---|
| CPU only | 4 GB free RAM; on a 2-core VM ≈ 1–2 min per document for all three modules; on an 8-core laptop typically well under a minute |
| GPU | install a CUDA/Metal build of llama-cpp-python and set `n_gpu_layers = -1`; seconds per document |
| Disk | ≈1.1 GB model + ≈0.1 GB embedding model + caches |

## 8. What was verified in the build environment

The build sandbox could not reach Hugging Face. The pipeline was therefore executed end-to-end with a
stand-in GGUF model (TinyLlama-1.1B-Chat, 2-bit IQ2_XXS quantisation, obtained from an npm data package,
SHA-256 `9ee11adf…1add0`). This verified that the LLM is really invoked, that grammar-constrained decoding
restricts IDs to the candidate list and that the validator rejects unsupported output — that 2-bit model
produced placeholder evidence and *all* its selections were correctly rejected. **The extraction quality of
the default Qwen2.5 model was not measured in the sandbox**; run the evaluation framework on your machine.
