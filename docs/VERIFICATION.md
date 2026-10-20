# Verification report (build environment, 20 Oct 2026)

Environment: Linux VM, 2 CPU cores, 7 GB RAM, no GPU, Python 3.13, torch 2.14, sentence-transformers 6.1,
llama-cpp-python 0.3.36 (built from source). Hugging Face, ESCO and Eurostat sites were **blocked** by the
sandbox network policy, so stand-in model files were used:

* embeddings: an fp16 repackaging of `all-MiniLM-L6-v2` (npm `@lat.md/embed-minilm-fp16`). Sanity check:
  the original endpoint `/extract-skills` returned exactly the two skill URIs shown in the original ESCOX README
  for the README's example sentence.
* LLM: TinyLlama-1.1B-Chat, 2-bit IQ2_XXS GGUF (npm `slaunt-model-tinyllama-1b-part1/2`, SHA-256
  `9ee11adf737960ff06123d5c4fba24d12c5b05ba460a7afa88950fbf2b61add0`). Far weaker than the default Qwen2.5-1.5B.

| Check | Result |
|---|---|
| Application starts (`python -m escox3`) | ✅ server up; GUI loads without console errors |
| Each module alone and all combinations | ✅ (GUI: sectors-only run; API: all three; disabled modules report `not_requested`) |
| No module selected / empty text | ✅ clear error messages (GUI and API, HTTP 400) |
| LLM actually invoked | ✅ 3 calls per hybrid analysis, latency 20–63 s each, token counts and raw output recorded |
| LLM output constrained to taxonomy | ✅ grammar-constrained decoding: every ID emitted was in the candidate enum; validator unit test rejects unknown IDs, duplicates, bad fields, unverifiable quotes |
| Unsupported output rejected | ✅ the 2-bit stand-in model returned placeholder evidence ("evidence", "..."); all such selections were rejected and listed as uncertain; modules reported "… not confidently identified" |
| Green skills mapped to supplied collection | ✅ every result resolves to a row of `greenSkillsCollection_en.csv` |
| Occupations use validated ISCO-08 | ✅ all 3 039 occupations have a 4-digit unit group present in the ISCO-08 list (unit test) |
| Sectors use validated NACE | ✅ only codes present in the crosswalk; hierarchy titles only when present in the data |
| Readable names instead of URIs | ✅ no URI in any result title (Playwright check); URIs only inside expanded details/exports |
| Multilingual NACE labels | ✅ Greek titles shown when EL selected |
| TXT / PDF / DOCX / pasted text | ✅ all extracted via API and GUI upload; unsupported type and non-PDF rejected with messages |
| JSON and CSV export | ✅ downloaded from GUI; CSV has labels + URIs + ISCO/NACE codes |
| Logo renders | ✅ SVG in header/footer, PNG + favicon generated |
| Original ESCOX functionality | ✅ `esco_skill_extractor/` unchanged (diff), `/extract-skills` and `/extract-occupations` served |
| Unit tests | ✅ 18 passed |
| Evaluation harness | ✅ ran A, A2, B, C + sensitivity sweep on a 2-document *software fixture* (numbers meaningless, not reported) |

## Not verified here

* **Extraction quality with the default Qwen2.5-1.5B-Instruct model** (or any capable LLM) — the model
  could not be downloaded. With the 2-bit stand-in, hybrid mode returned no accepted items, which shows the
  safeguards work but says nothing about quality. Run a hybrid analysis on your machine first.
* Automatic Hugging Face download path (`Llama` from `hf_hub_download`) and the `transformers` backend —
  code paths exist but were not executed; the OpenAI-compatible backend was tested against a local stub.
* GPU execution.
* Green occupation status — needs a reference file that was not supplied (see DATA_AND_GAPS.md).
* Original `/extract-occupations` returned `[]` for the README example sentence while the original README
  shows several occupations; skills matched exactly. Likely a data/threshold difference in the README
  example; behaviour of the unchanged original code was not altered.
