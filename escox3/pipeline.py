"""
ESCOX 3.0 analysis pipeline.

    Unstructured text
      -> fragments (sentence + clause level, with offsets)
      -> Stage 1  semantic candidate retrieval (original ESCOX embeddings, top-k)
      -> Stage 2  LLM contextual reasoning over (text, candidates)
      -> Stage 3  taxonomy-constrained JSON (grammar: ids restricted to candidates)
      -> Stage 4  validation against the reference store + evidence verification
      -> structured green skills / occupations (ISCO-08) / sectors (NACE Rev. 2.1)

Modes (``mode``):
  * "hybrid"    – Stages 1-4 (ESCOX 3.0 default; Experiment C)
  * "retrieval" – Stage 1 only with acceptance thresholds (Experiment A-style)
  * "llm_only"  – LLM extraction without retrieval, mapped to the taxonomy by exact label/code
                  lookup (Experiment B)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from .config import CACHE_DIR, Config
from .llm import NullBackend, load_backend
from .llm.prompts import (PROMPT_VERSION, build_open_messages, build_selection_messages, open_schema,
                          selection_schema)
from .retrieval import Candidate, Chunk, Embedder, EmbeddingIndex, chunk_text
from .taxonomy import ReferenceStore
from .validation import ValidationReport, parse_json_object, validate_open_items, validate_selection, verify_evidence

log = logging.getLogger("escox3.pipeline")

MODULES = ("green_skills", "green_occupations", "sectors")
MODES = ("hybrid", "retrieval", "llm_only")
MAX_ITEMS = {"green_skills": 15, "green_occupations": 3, "sectors": 3}
ID_PREFIX = {"green_skills": "S", "green_occupations": "O", "sectors": "N"}
NONE_FOUND = {
    "green_skills": "No green skill from the reference collection was confidently identified.",
    "green_occupations": "No occupation confidently identified.",
    "sectors": "Sector not confidently identified.",
}


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


class ESCOX3:
    """Main entry point. Load once, then call :meth:`analyse` for each document."""

    def __init__(self, config: Optional[Config] = None, load_llm: bool = True, load_legacy: bool = True,
                 llm_backend=None):
        self.cfg = config or Config()
        import torch

        self.device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.store = ReferenceStore(green_share_threshold=self.cfg.green_share_threshold)

        # ---- reuse the original ESCOX extractor (model + cached ESCO embeddings + legacy API)
        self.legacy = None
        if load_legacy:
            self.legacy = self._load_legacy()
            st_model = self.legacy._model
        else:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(self.cfg.embedding_model, device=self.device)
        self.embedder = Embedder(st_model, self.cfg.embedding_model, self.device)

        gs, occ, nace = self.store.green_skills, self.store.occupations, self.store.nace
        self.skill_index = self.embedder.build_index("green_skills", gs.uri.tolist(), gs.embed_text.tolist())
        self.occ_index = self._occupation_index(occ)
        self.nace_index = self.embedder.build_index("nace", nace.code.tolist(), nace.title_en.tolist())

        # label lookups for LLM-only mode (exact preferred / alternative labels from the supplied files)
        self._skill_by_label: Dict[str, str] = {}
        for uri, lab, alts in zip(gs.uri, gs.label, gs.alt_labels):
            for l in [lab] + [a for a in alts.split(" | ") if a]:
                self._skill_by_label.setdefault(_norm_label(l), uri)
        self._occ_by_label = {_norm_label(l): u for u, l in zip(occ.uri, occ.label)}
        self._nace_by_title = {_norm_label(t): c for c, t in zip(nace.code, nace.title_en)}

        self.llm = llm_backend if llm_backend is not None else (
            load_backend(self.cfg.llm, self.device) if load_llm else NullBackend("LLM not loaded"))
        log.info("ESCOX 3.0 ready (device=%s, llm=%s)", self.device, self.llm.info().get("model_name"))

    # ------------------------------------------------------------------ setup helpers
    def _load_legacy(self):
        from esco_skill_extractor import SkillExtractor

        # The original caches embeddings without recording the model; invalidate when the model changes.
        marker = CACHE_DIR / "legacy_embedding_model.txt"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if marker.exists() and marker.read_text().strip() != self.cfg.embedding_model:
            log.info("Embedding model changed - removing stale original-ESCOX embeddings")
            SkillExtractor.remove_embeddings()
        ex = SkillExtractor(model=self.cfg.embedding_model, device=self.device)
        marker.write_text(self.cfg.embedding_model)
        return ex

    def _occupation_index(self, occ) -> EmbeddingIndex:
        if self.legacy is not None:
            pos = {u: i for i, u in enumerate(self.legacy._occupation_ids)}
            if all(u in pos for u in occ.uri):
                emb = self.legacy._occupation_embeddings.detach().cpu().numpy()[[pos[u] for u in occ.uri]]
                log.info("Reusing original ESCOX occupation embeddings for %d ESCO occupations", len(occ))
                return EmbeddingIndex("occupations", occ.uri.tolist(), emb)
        return self.embedder.build_index("occupations", occ.uri.tolist(), occ.embed_text.tolist())

    # ------------------------------------------------------------------ public API
    def analyse(self, text: str, extract_green_skills: bool = True, extract_green_occupations: bool = True,
                extract_sectors: bool = True, mode: Optional[str] = None,
                label_language: Optional[str] = None) -> dict:
        t_start = time.time()
        text = (text or "").replace("\r\n", "\n").strip()
        if not text:
            raise ValueError("The text is empty.")
        modules = [m for m, on in zip(MODULES, (extract_green_skills, extract_green_occupations, extract_sectors)) if on]
        if not modules:
            raise ValueError("Select at least one extraction module.")
        mode = (mode or self.cfg.mode).lower()
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Use one of {MODES}.")
        lang = (label_language or self.cfg.label_language or "en").lower()

        warnings: List[str] = []
        if mode in ("hybrid", "llm_only") and not self.llm.available:
            warnings.append(f"LLM unavailable ({self.llm.info().get('reason')}); results are semantic-retrieval "
                            "only and were NOT validated by an LLM.")
            mode = "retrieval"

        timings: Dict[str, float] = {}
        ctx = {"text": text, "calls": [], "lang": lang, "warnings": warnings}
        result = {"green_skills": [], "green_occupations": [], "sectors": [], "module_status": {},
                  "rejected_candidates": {}}

        if mode != "llm_only":
            t = time.time()
            r = self.cfg.retrieval
            chunks = chunk_text(text, r.min_chunk_chars, r.max_chunks)
            chunks.append(Chunk(text[:2000], 0, min(len(text), 2000), "document"))
            ctx["chunks"] = chunks
            ctx["q"] = self.embedder.encode([c.text for c in chunks])
            timings["fragmenting_and_embedding_s"] = round(time.time() - t, 3)

        occ_cands = None
        for module in modules:
            t = time.time()
            if mode == "llm_only":
                items, rejected = self._llm_only(module, ctx)
            elif module == "green_skills":
                items, rejected = self._skills(ctx, mode)
            elif module == "green_occupations":
                occ_cands = occ_cands or self._occupation_candidates(ctx)
                items, rejected = self._occupations(ctx, mode, occ_cands)
            else:
                items, rejected = self._sectors(ctx, mode, result.get("green_occupations"), occ_cands)
            timings[f"{module}_s"] = round(time.time() - t, 3)
            for i, it in enumerate(items, 1):
                it["rank"] = i
            result[module] = items
            result["rejected_candidates"][module] = rejected
            result["module_status"][module] = {"status": "ok" if items else "none_found",
                                               "message": None if items else NONE_FOUND[module],
                                               "n_results": len(items)}
        for m in MODULES:
            if m not in modules:  # disabled modules: nothing is computed for them
                result["module_status"][m] = {"status": "not_requested", "message": None, "n_results": 0}

        llm_info = self.llm.info()
        total = round(time.time() - t_start, 3)
        result["metadata"] = {
            "escox_version": __version__,
            "model": llm_info.get("model_name") if mode != "retrieval" else None,
            "processing_time": total,
            "mode": mode,
            "modules_requested": modules,
            "llm_invoked": any(c.get("invoked") for c in ctx["calls"]),
            "llm_calls": ctx["calls"],
            "prompt_version": PROMPT_VERSION,
            "models": {"embedding": {"name": self.cfg.embedding_model, "library": "sentence-transformers",
                                     "device": self.device,
                                     "reused_from": "original ESCOX SkillExtractor" if self.legacy else None},
                       "llm": llm_info},
            "timings": timings,
            "text": {"chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "fragments": len(ctx.get("chunks", [])) - 1 if ctx.get("chunks") else None},
            "label_language": lang,
            "taxonomy": self.store.manifest.get("taxonomy_versions"),
            "green_occupation_reference": self.store.summary()["green_occupation_reference"],
            "retrieval_config": vars(self.cfg.retrieval),
            "warnings": warnings,
        }
        return result

    # ------------------------------------------------------------------ module: green skills
    def _skills(self, ctx, mode) -> Tuple[List[dict], List[dict]]:
        r = self.cfg.retrieval
        cands = self.skill_index.search(ctx["q"], r.skill_candidate_threshold, r.skill_top_k_per_chunk,
                                        r.skill_max_candidates)

        def describe(uri):
            s = self.store.skill(uri)
            return f"{s['label']} | {s['description'][:140]}"

        def build(uri):
            return dict(self.store.skill(uri))

        return self._resolve(ctx, mode, "green_skills", cands, describe, build, r.skill_accept_threshold, None)

    # ------------------------------------------------------------------ module: occupations
    def _occupation_candidates(self, ctx) -> List[Candidate]:
        r = self.cfg.retrieval
        return self.occ_index.search(ctx["q"], r.occupation_candidate_threshold, r.occupation_top_k_per_chunk,
                                     r.occupation_max_candidates)

    def _occupations(self, ctx, mode, cands) -> Tuple[List[dict], List[dict]]:
        r = self.cfg.retrieval

        def describe(uri):
            o = self.store.occupation(uri)
            return f"{o['label']} | {o['isco_code'] or '-'} {o['isco_label'] or ''} | {o['description'][:130]}"

        def build(uri):
            return dict(self.store.occupation(uri))

        return self._resolve(ctx, mode, "green_occupations", cands, describe, build,
                             r.occupation_accept_threshold, r.occupation_retrieval_only_max)

    # ------------------------------------------------------------------ module: sectors
    def _sectors(self, ctx, mode, accepted_occupations, occ_cands) -> Tuple[List[dict], List[dict]]:
        r = self.cfg.retrieval
        cands = self.nace_index.search(ctx["q"], r.sector_candidate_threshold, r.sector_top_k_per_chunk,
                                       r.sector_max_candidates)
        if mode == "hybrid" and r.sector_use_occupation_crosswalk:
            # Occupation != sector: crosswalk codes only become *candidates*; the LLM decides from the
            # economic activity described in the text.
            if accepted_occupations:
                occ_uris = [o["esco_uri"] for o in accepted_occupations]
            else:
                occ_uris = [c.id for c in (occ_cands or self._occupation_candidates(ctx))]
            occ_uris = occ_uris[: r.sector_crosswalk_occupations]
            have = {c.id: c for c in cands}
            for ou in occ_uris:
                for code in self.store.occupation(ou)["nace_codes"]:
                    if code in have:
                        if "occupation_crosswalk" not in have[code].sources:
                            have[code].sources.append("occupation_crosswalk")
                        continue
                    s, j = self.nace_index.score_of(ctx["q"], code)
                    c = Candidate(code, s, j, sources=["occupation_crosswalk"])
                    cands.append(c)
                    have[code] = c
            cands = sorted(cands, key=lambda c: -c.score)[: r.sector_max_candidates + 5]
            for i, c in enumerate(cands, 1):
                c.rank = i

        lang = ctx["lang"]

        def describe(code):
            return f"{code} | {self.store.nace_title(code, 'en')}"

        def build(code):
            return dict(self.store.nace_entry(code, lang))

        items, rejected = self._resolve(ctx, mode, "sectors", cands, describe, build,
                                        r.sector_accept_threshold, r.sector_retrieval_only_max)
        # keep the most specific code when both a parent and its child were selected
        codes = {i["nace_code"] for i in items}
        kept = []
        for it in items:
            if any(o != it["nace_code"] and o.startswith(it["nace_code"]) for o in codes):
                rejected.append({"id": it["nace_code"], "label": it["label"], "similarity": it.get("similarity"),
                                 "reason": "parent of a more specific selected code (shown in its hierarchy)"})
            else:
                kept.append(it)
        return kept, rejected

    # ------------------------------------------------------------------ shared resolution logic
    def _resolve(self, ctx, mode, module, cands: List[Candidate], describe: Callable[[str], str],
                 build: Callable[[str], dict], accept_thr: float, retrieval_max: Optional[int]):
        chunks: List[Chunk] = ctx["chunks"]

        def base(c: Candidate) -> dict:
            item = build(c.id)
            ch = chunks[c.chunk_index]
            item.update({"similarity": round(c.score, 4), "retrieval_rank": c.rank,
                         "matched_fragment": ch.text if ch.kind != "document" else None,
                         "candidate_source": list(c.sources)})
            return item

        if mode == "retrieval":
            acc = [c for c in cands if c.score >= accept_thr and "semantic" in c.sources]
            if retrieval_max:
                acc = acc[:retrieval_max]
            items = []
            for c in acc:
                it = base(c)
                ch = chunks[c.chunk_index]
                it.update({"support": "semantic_similarity", "evidence": it["matched_fragment"],
                           "evidence_verified": ch.kind != "document",
                           "evidence_span": [ch.start, ch.end] if ch.kind != "document" else None,
                           "llm_decision": "not_applied",
                           "confidence": "high" if c.score >= accept_thr + 0.15 else "medium",
                           "method": "semantic retrieval (cosine similarity >= %.2f)" % accept_thr})
                items.append(it)
            rejected = [{"id": c.id, "label": self.store.resolve_label(c.id, ctx["lang"])["label"],
                         "similarity": round(c.score, 4), "reason": f"similarity below acceptance threshold {accept_thr}"}
                        for c in cands if c not in acc][:10]
            return items, rejected

        # ---- hybrid: LLM validation of the retrieved candidates
        if not cands:
            ctx["calls"].append({"module": module, "invoked": False, "reason": "no candidates above retrieval threshold"})
            return [], []
        sel, rep, call = self._llm_select(ctx, module, cands, describe)
        ctx["calls"].append(call)
        by_sid = {f"{ID_PREFIX[module]}{i}": c for i, c in enumerate(cands, 1)}
        items = []
        for a in sel:
            c = by_sid[a["id"]]
            it = base(c)
            it.update({"support": a["support"], "evidence": a["evidence"], "evidence_verified": a["evidence_verified"],
                       "evidence_span": a["evidence_span"], "evidence_match_score": a["evidence_match_score"],
                       "llm_decision": "accepted", "llm_order": a["llm_order"],
                       "confidence": self._confidence(a["support"], a["evidence_verified"]),
                       "method": "semantic retrieval + LLM validation"})
            if a.get("note"):
                it["validation_note"] = a["note"]
            items.append(it)
        if module != "green_skills":
            items.sort(key=lambda x: x["llm_order"])
        else:
            items.sort(key=lambda x: (-{"high": 2, "medium": 1, "low": 0}[x["confidence"]], -x["similarity"]))
        accepted_ids = {a["id"] for a in sel} | {r.get("id") for r in rep.rejected}
        rejected = []
        for sid, c in by_sid.items():
            if sid not in accepted_ids:
                rejected.append({"id": c.id, "label": self.store.resolve_label(c.id, ctx["lang"])["label"],
                                 "similarity": round(c.score, 4),
                                 "reason": "rejected by LLM (not supported by the text)" if rep.valid_json
                                 else f"LLM output invalid ({rep.error}); candidate not validated"})
        for r in rep.rejected:
            c = by_sid.get(r.get("id"))
            rejected.insert(0, {"id": c.id if c else r.get("id"),
                                "label": self.store.resolve_label(c.id, ctx["lang"])["label"] if c else None,
                                "similarity": round(c.score, 4) if c else None, "reason": r["reason"],
                                "llm_evidence": r.get("evidence")})
        return items, rejected

    @staticmethod
    def _confidence(support: str, verified: bool) -> str:
        if verified and support == "explicit":
            return "high"
        if verified:
            return "medium"
        return "low"

    # ------------------------------------------------------------------ LLM calls
    def _condense(self, ctx, cands: List[Candidate]) -> str:
        text, limit = ctx["text"], self.cfg.llm.max_text_chars
        if len(text) <= limit:
            return text
        head = text[: int(limit * 0.55)]
        chunks = ctx.get("chunks", [])
        ex, used = [], len(head)
        for c in cands:
            ch = chunks[c.chunk_index] if c.chunk_index < len(chunks) else None
            if ch is None or ch.kind == "document" or ch.start < len(head):
                continue
            # expand clause to its enclosing line for context
            s = text.rfind("\n", 0, ch.start) + 1
            e = text.find("\n", ch.end)
            frag = text[s: e if e != -1 else len(text)].strip()[:400]
            if frag and frag not in ex and used + len(frag) < limit:
                ex.append(frag)
                used += len(frag)
        msg = (f"Input longer than {limit} characters: the LLM received the beginning of the text plus the "
               "retrieved evidence excerpts (evidence is still verified against the full text).")
        if msg not in ctx["warnings"]:
            ctx["warnings"].append(msg)
        return head + "\n[...]\nRelevant excerpts:\n" + "\n".join(f"- {e}" for e in ex)

    def _fit_prompt(self, module, llm_text, lines, max_tokens):
        """Shrink candidate descriptions / candidates / text until the prompt fits the context window."""
        adjustments = []
        counter = getattr(self.llm, "count_tokens", None)
        n_ctx = getattr(self.llm, "n_ctx", None)
        msgs = build_selection_messages(module, llm_text, lines)
        if not counter or not n_ctx:
            return msgs, lines, adjustments
        budget = n_ctx - max_tokens - 16
        while counter(msgs) > budget:
            if any(len(l) > 90 for l in lines):
                lines = [l[:90] for l in lines]
                adjustments.append("candidate descriptions shortened")
            elif len(lines) > 5:
                lines = lines[:-1]
                adjustments.append("lowest-ranked candidate dropped")
            elif len(llm_text) > 400:
                llm_text = llm_text[: int(len(llm_text) * 0.8)]
                adjustments.append("text shortened")
            else:
                break
            msgs = build_selection_messages(module, llm_text, lines)
        return msgs, lines, sorted(set(adjustments))

    def _llm_select(self, ctx, module, cands, describe):
        sids = [f"{ID_PREFIX[module]}{i}" for i in range(1, len(cands) + 1)]
        lines = [f"{sid} | {describe(c.id)}" for sid, c in zip(sids, cands)]
        max_items = MAX_ITEMS[module]
        max_tokens = min(self.cfg.llm.max_new_tokens, 60 + 70 * max_items)
        llm_text = self._condense(ctx, cands)
        msgs, lines, adj = self._fit_prompt(module, llm_text, lines, max_tokens)
        sids = sids[: len(lines)]
        schema = selection_schema(sids, max_items, self.cfg.llm.evidence_max_chars)
        resp = self.llm.generate(msgs, schema=schema, max_tokens=max_tokens)
        parsed, perr = parse_json_object(resp.text) if not resp.error else (None, resp.error)
        rep = (validate_selection(parsed, sids, ctx["text"], max_items, self.cfg.llm.evidence_max_chars,
                                  self.cfg.llm.require_verified_evidence)
               if parsed is not None else ValidationReport(error=perr))
        call = {"module": module, "invoked": True, "n_candidates": len(sids), "latency_s": resp.latency_s,
                "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                "grammar_constrained": resp.constrained, "valid_json": rep.valid_json,
                "validation": rep.to_dict(), "prompt_adjustments": adj, "error": resp.error}
        if self.cfg.llm.store_raw_output:
            call["raw_output"] = resp.text
        return rep.accepted, rep, call

    # ------------------------------------------------------------------ Experiment B: LLM only
    def _llm_only(self, module, ctx) -> Tuple[List[dict], List[dict]]:
        text = ctx["text"][: self.cfg.llm.max_text_chars]
        max_items = MAX_ITEMS[module]
        resp = self.llm.generate(build_open_messages(module, text), schema=open_schema(max_items),
                                 max_tokens=min(self.cfg.llm.max_new_tokens, 60 + 80 * max_items))
        parsed, perr = parse_json_object(resp.text) if not resp.error else (None, resp.error)
        raw, serr = validate_open_items(parsed) if parsed is not None else ([], perr)
        items, rejected, seen = [], [], set()
        for x in raw[:max_items]:
            label, code = str(x.get("label", "")), str(x.get("code", "")).strip().upper().replace(" ", "")
            ref = None
            if module == "green_skills":
                ref = self._skill_by_label.get(_norm_label(label))
            elif module == "green_occupations":
                ref = self._occ_by_label.get(_norm_label(label))
            else:
                ref = code if self.store.nace_entry(code) else self._nace_by_title.get(_norm_label(label))
            if not ref:
                rejected.append({"id": None, "label": label, "similarity": None,
                                 "reason": "label/code not found in the reference taxonomy (invalid or hallucinated)"})
                continue
            if ref in seen:
                continue
            seen.add(ref)
            it = (self.store.skill(ref) if module == "green_skills" else
                  self.store.occupation(ref) if module == "green_occupations" else self.store.nace_entry(ref, ctx["lang"]))
            it = dict(it)
            sup = str(x.get("support", "implicit")).lower()
            sup = sup if sup in ("explicit", "implicit") else "implicit"
            ok, span, score = verify_evidence(str(x.get("evidence", "")), ctx["text"])
            if self.cfg.llm.require_verified_evidence and not ok:
                rejected.append({"id": ref, "label": label, "similarity": None,
                                 "reason": "unsupported: evidence quote not found in the text"})
                continue
            if sup == "explicit" and not ok:
                sup = "implicit"
            it.update({"support": sup, "evidence": x.get("evidence"), "evidence_verified": ok, "evidence_span": span,
                       "similarity": None, "retrieval_rank": None, "llm_decision": "generated",
                       "confidence": self._confidence(sup, ok), "candidate_source": ["llm_generation"],
                       "method": "LLM-only extraction + exact taxonomy lookup"})
            items.append(it)
        ctx["calls"].append({"module": module, "invoked": True, "latency_s": resp.latency_s,
                             "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                             "grammar_constrained": resp.constrained, "valid_json": serr is None,
                             "validation": {"raw_items": len(raw), "accepted": len(items),
                                            "invalid_labels": sum(1 for r in rejected if "taxonomy" in r["reason"]),
                                            "unsupported": sum(1 for r in rejected if "unsupported" in r["reason"]),
                                            "error": serr},
                             "error": resp.error, **({"raw_output": resp.text} if self.cfg.llm.store_raw_output else {})})
        return items, rejected

    # ------------------------------------------------------------------ legacy (original ESCOX) API
    def legacy_skills(self, texts: List[str]) -> List[List[str]]:
        return self.legacy.get_skills(texts) if self.legacy else [[] for _ in texts]

    def legacy_occupations(self, texts: List[str]) -> List[List[str]]:
        return self.legacy.get_occupations(texts) if self.legacy else [[] for _ in texts]

    def original_escox_match(self, text: str, module: str) -> List[str]:
        """Experiment A baseline: the unmodified original ESCOX matching algorithm
        (original fragment splitter, max-similarity per fragment, fixed threshold) applied to the
        ESCOX 3.0 reference sets. Requires the legacy extractor."""
        import torch

        if self.legacy is None:
            raise RuntimeError("Experiment A baseline needs load_legacy=True")
        idx, thr = {"green_skills": (self.skill_index, self.legacy.skills_threshold),
                    "green_occupations": (self.occ_index, self.legacy.occupation_threshold),
                    "sectors": (self.nace_index, self.cfg.retrieval.sector_accept_threshold)}[module]
        emb = torch.tensor(idx.emb, device=self.device)
        return self.legacy._get_entity([text], np.array(idx.ids), emb, thr)[0]
