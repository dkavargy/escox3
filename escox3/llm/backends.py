"""
LLM inference backends for ESCOX 3.0 (Stage 2 — contextual reasoning).

* ``llama_cpp``    – default. Local GGUF model through llama-cpp-python. Supports
                     JSON-schema *grammar-constrained decoding*: the candidate IDs are
                     an ``enum`` so the model physically cannot emit an ID that is not
                     in the retrieved candidate list.
* ``transformers`` – local Hugging Face model (greedy decoding, prompt-constrained,
                     post-hoc validated).
* ``openai``       – any OpenAI-compatible chat-completions server (Ollama, vLLM,
                     LM Studio, llama.cpp server, OpenAI). Optional; requests a JSON
                     schema response format when the server supports it.
* ``none``         – LLM disabled (retrieval-only mode).

Every call returns an :class:`LLMResponse` carrying latency and token counts so the
metadata of each analysis documents exactly how the LLM was used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import CACHE_DIR, LLMConfig

log = logging.getLogger("escox3.llm")


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    constrained: bool = False
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMBackend:
    name = "base"
    available = False

    def info(self) -> Dict[str, Any]:
        return {"backend": self.name, "available": self.available}

    def generate(self, messages: List[Dict[str, str]], schema: Optional[dict] = None,
                 max_tokens: Optional[int] = None) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class NullBackend(LLMBackend):
    name = "none"
    available = False

    def __init__(self, reason: str = "LLM disabled by configuration"):
        self.reason = reason

    def info(self):
        return {"backend": "none", "available": False, "reason": self.reason}

    def generate(self, messages, schema=None, max_tokens=None):
        return LLMResponse("", 0.0, error=self.reason)


def _file_sha256(path: str) -> str:
    """SHA-256 of a model file, cached by (path, size, mtime) — model files are large."""
    p = Path(path)
    st = p.stat()
    cache = CACHE_DIR / "model_hashes.json"
    key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    try:
        known = json.loads(cache.read_text()) if cache.exists() else {}
    except Exception:
        known = {}
    if key in known:
        return known[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    known[key] = h.hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, indent=1))
    return known[key]


# --------------------------------------------------------------------------- llama.cpp
class LlamaCppBackend(LLMBackend):
    name = "llama_cpp"

    def __init__(self, cfg: LLMConfig):
        from llama_cpp import Llama, __version__ as llama_cpp_version  # noqa: F401

        self.cfg = cfg
        n_threads = cfg.n_threads or max(1, (os.cpu_count() or 2))
        t0 = time.time()
        if cfg.model_path:
            self.model_file = cfg.model_path
            self.source = f"local file {Path(cfg.model_path).name}"
        else:
            # Downloaded once from the Hugging Face Hub into the local HF cache.
            from huggingface_hub import hf_hub_download

            self.model_file = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.hf_filename)
            self.source = f"huggingface.co/{cfg.hf_repo_id} ({cfg.hf_filename})"
        # read GGUF metadata first (vocab only) so the context never exceeds the trained context length
        probe = Llama(model_path=self.model_file, vocab_only=True, verbose=False)
        meta = dict(getattr(probe, "metadata", {}) or {})
        del probe
        arch = meta.get("general.architecture", "")
        trained = meta.get(f"{arch}.context_length")
        self.n_ctx = min(cfg.n_ctx, int(trained)) if trained else cfg.n_ctx
        self.llm = Llama(model_path=self.model_file, n_ctx=self.n_ctx, n_threads=n_threads,
                         n_gpu_layers=cfg.n_gpu_layers, seed=cfg.seed, verbose=False)
        self.load_time_s = round(time.time() - t0, 2)
        self.metadata = {k: meta[k] for k in ("general.name", "general.basename", "general.architecture",
                                               "general.size_label", "general.license", "general.file_type",
                                               "general.quantization_version", f"{arch}.context_length")
                         if k in meta}
        self.llama_cpp_version = llama_cpp_version
        self.sha256 = _file_sha256(self.model_file)
        self.n_threads = n_threads
        self._lock = threading.Lock()
        self.available = True
        log.info("LLM loaded: %s (%s) in %.1fs", self.metadata.get("general.name"), self.source, self.load_time_s)

    def info(self):
        return {
            "backend": self.name, "available": True,
            "model_name": self.metadata.get("general.name"),
            "model_source": self.source,
            "model_file": Path(self.model_file).name,
            "model_sha256": self.sha256,
            "gguf_metadata": self.metadata,
            "runtime": f"llama-cpp-python {self.llama_cpp_version}",
            "usage": "inference only (pretrained instruction-tuned model, no fine-tuning by ESCOX)",
            "decoding": {"temperature": self.cfg.temperature, "top_p": self.cfg.top_p, "seed": self.cfg.seed,
                         "max_new_tokens": self.cfg.max_new_tokens, "n_ctx": self.n_ctx,
                         "grammar_constrained_json": self.cfg.constrained_decoding},
            "threads": self.n_threads, "gpu_layers": self.cfg.n_gpu_layers, "load_time_s": self.load_time_s,
        }

    def count_tokens(self, messages) -> int:
        text = "\n".join(m["content"] for m in messages)
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=True)) + 12 * len(messages)

    def generate(self, messages, schema=None, max_tokens=None):
        kwargs = dict(messages=messages, temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                      max_tokens=max_tokens or self.cfg.max_new_tokens, seed=self.cfg.seed)
        constrained = bool(schema and self.cfg.constrained_decoding)
        if constrained:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}
        t0 = time.time()
        with self._lock:
            try:
                self.llm.reset()  # deterministic: no KV-cache reuse across requests
                out = self.llm.create_chat_completion(**kwargs)
            except Exception as e:  # pragma: no cover - runtime failure path
                log.exception("llama.cpp generation failed")
                return LLMResponse("", time.time() - t0, error=f"{type(e).__name__}: {e}", constrained=constrained)
        usage = out.get("usage", {})
        return LLMResponse(out["choices"][0]["message"]["content"] or "", round(time.time() - t0, 3),
                           usage.get("prompt_tokens"), usage.get("completion_tokens"), constrained,
                           extra={"finish_reason": out["choices"][0].get("finish_reason")})


# --------------------------------------------------------------------------- transformers
class TransformersBackend(LLMBackend):
    name = "transformers"

    def __init__(self, cfg: LLMConfig, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tv

        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(cfg.transformers_model)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(cfg.transformers_model, dtype=dtype).to(self.device)
        self.model.eval()
        self.load_time_s = round(time.time() - t0, 2)
        self.tv = tv
        self.n_ctx = min(cfg.n_ctx, int(getattr(self.model.config, "max_position_embeddings", cfg.n_ctx)))
        self._lock = threading.Lock()
        self.available = True

    def info(self):
        return {"backend": self.name, "available": True, "model_name": self.cfg.transformers_model,
                "model_source": self.cfg.transformers_model, "runtime": f"transformers {self.tv}",
                "usage": "inference only (pretrained instruction-tuned model, no fine-tuning by ESCOX)",
                "decoding": {"greedy": True, "max_new_tokens": self.cfg.max_new_tokens,
                             "grammar_constrained_json": False},
                "device": self.device, "load_time_s": self.load_time_s}

    def count_tokens(self, messages) -> int:
        return len(self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def generate(self, messages, schema=None, max_tokens=None):
        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with self._lock, torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_tokens or self.cfg.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        gen = out[0, ids["input_ids"].shape[1]:]
        return LLMResponse(self.tok.decode(gen, skip_special_tokens=True), round(time.time() - t0, 3),
                           int(ids["input_ids"].shape[1]), int(gen.shape[0]), False)


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatibleBackend(LLMBackend):
    name = "openai"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.url = cfg.api_base.rstrip("/") + "/chat/completions"
        self.key = os.environ.get(cfg.api_key_env, "")
        self.schema_supported = True
        self.available = True

    def info(self):
        return {"backend": self.name, "available": True, "model_name": self.cfg.api_model,
                "model_source": self.cfg.api_base, "usage": "inference only via API",
                "decoding": {"temperature": self.cfg.temperature, "top_p": self.cfg.top_p, "seed": self.cfg.seed,
                             "max_new_tokens": self.cfg.max_new_tokens,
                             "json_schema_response_format": self.schema_supported}}

    def _post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.cfg.api_timeout_s) as r:
            return json.loads(r.read().decode())

    def generate(self, messages, schema=None, max_tokens=None):
        body = {"model": self.cfg.api_model, "messages": messages, "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p, "seed": self.cfg.seed, "max_tokens": max_tokens or self.cfg.max_new_tokens}
        constrained = False
        if schema and self.cfg.constrained_decoding and self.schema_supported:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "escox_selection", "schema": schema}}
            constrained = True
        t0 = time.time()
        try:
            try:
                out = self._post(body)
            except urllib.error.HTTPError as e:
                if constrained and e.code in (400, 422):
                    self.schema_supported = False
                    body["response_format"] = {"type": "json_object"}
                    constrained = False
                    out = self._post(body)
                else:
                    raise
        except Exception as e:
            return LLMResponse("", time.time() - t0, error=f"{type(e).__name__}: {e}")
        usage = out.get("usage", {}) or {}
        return LLMResponse(out["choices"][0]["message"].get("content") or "", round(time.time() - t0, 3),
                           usage.get("prompt_tokens"), usage.get("completion_tokens"), constrained)


def load_backend(cfg: LLMConfig, device: Optional[str] = None) -> LLMBackend:
    b = (cfg.backend or "none").lower()
    if b in ("none", "off", "disabled"):
        return NullBackend()
    try:
        if b in ("llama_cpp", "llamacpp", "gguf"):
            return LlamaCppBackend(cfg)
        if b in ("transformers", "hf"):
            return TransformersBackend(cfg, device)
        if b in ("openai", "ollama", "api"):
            return OpenAICompatibleBackend(cfg)
    except Exception as e:
        log.error("Could not load LLM backend %s: %s", b, e)
        return NullBackend(f"LLM backend '{b}' failed to load: {type(e).__name__}: {e}")
    return NullBackend(f"Unknown LLM backend '{b}'")
