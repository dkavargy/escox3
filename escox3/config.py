"""
ESCOX 3.0 configuration.

All tunable parameters live in one dataclass so that every experiment can be
reproduced from a single serialisable configuration (see ``Config.to_dict``).
Values can be overridden from a TOML file (``--config path.toml``) or from
environment variables prefixed with ``ESCOX_`` (e.g. ``ESCOX_LLM_BACKEND=none``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class RetrievalConfig:
    # Candidate-retrieval stage (Stage 1). Scores are cosine similarities in [-1, 1].
    skill_candidate_threshold: float = 0.40      # minimum similarity to become an LLM candidate
    skill_accept_threshold: float = 0.60         # retrieval-only acceptance (= original ESCOX skill threshold)
    skill_top_k_per_chunk: int = 5
    skill_max_candidates: int = 20

    occupation_candidate_threshold: float = 0.35
    occupation_accept_threshold: float = 0.55    # retrieval-only acceptance (= original ESCOX occupation threshold)
    occupation_top_k_per_chunk: int = 5
    occupation_max_candidates: int = 15
    occupation_retrieval_only_max: int = 5       # max occupations returned in retrieval-only mode

    sector_candidate_threshold: float = 0.30
    sector_accept_threshold: float = 0.50        # retrieval-only acceptance
    sector_top_k_per_chunk: int = 5
    sector_max_candidates: int = 15
    sector_retrieval_only_max: int = 3
    sector_use_occupation_crosswalk: bool = True  # add NACE codes linked (ESCO-NACE crosswalk) to top occupation candidates
    sector_crosswalk_occupations: int = 3         # how many top occupation candidates feed the crosswalk

    min_chunk_chars: int = 4
    max_chunks: int = 400


@dataclass
class LLMConfig:
    # backend: "llama_cpp" (default, local GGUF), "transformers" (local HF), "openai" (any
    # OpenAI-compatible server: Ollama, vLLM, LM Studio, OpenAI ...), or "none" (disable LLM).
    backend: str = "llama_cpp"
    # llama_cpp: either a local GGUF path or a Hugging Face repo + filename (auto-download).
    model_path: Optional[str] = None
    hf_repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    hf_filename: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    # transformers backend model id (or local directory)
    transformers_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # openai-compatible backend
    api_base: str = "http://localhost:11434/v1"   # Ollama default
    api_model: str = "qwen2.5:1.5b-instruct"
    api_key_env: str = "ESCOX_LLM_API_KEY"
    api_timeout_s: int = 120

    # Decoding — deterministic by default for reproducibility.
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_new_tokens: int = 700
    n_ctx: int = 8192
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0             # set -1 to offload all layers when a GPU build of llama.cpp is installed
    constrained_decoding: bool = True # JSON-schema grammar with an enum of candidate IDs (llama_cpp / openai)
    max_text_chars: int = 6000        # longer inputs are condensed (head + retrieval evidence) before prompting
    evidence_max_chars: int = 200
    require_verified_evidence: bool = True  # reject LLM selections whose evidence quote is not found in the text
    store_raw_output: bool = True     # keep raw LLM output in metadata for auditability


@dataclass
class Config:
    embedding_model: str = "all-MiniLM-L6-v2"   # same default as original ESCOX
    device: Optional[str] = None                 # "cuda" if available else "cpu"
    mode: str = "hybrid"                         # "hybrid" | "retrieval" | "llm_only"
    label_language: str = "en"
    green_share_threshold: float = 0.10          # only used if a green-share reference was imported (see docs)
    host: str = "localhost"
    port: int = 8000
    max_upload_mb: int = 15
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Optional[str | os.PathLike] = None) -> "Config":
        cfg = cls()
        if path:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            cfg._apply(data)
        cfg._apply_env()
        return cfg

    def _apply(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if key == "retrieval" and isinstance(value, dict):
                _apply_dc(self.retrieval, value)
            elif key == "llm" and isinstance(value, dict):
                _apply_dc(self.llm, value)
            elif hasattr(self, key):
                setattr(self, key, value)

    def _apply_env(self) -> None:
        for f in fields(self):
            env = os.environ.get(f"ESCOX_{f.name.upper()}")
            if env is not None and f.name not in ("retrieval", "llm"):
                setattr(self, f.name, _coerce(env, getattr(self, f.name)))
        for prefix, obj in (("ESCOX_RETRIEVAL_", self.retrieval), ("ESCOX_LLM_", self.llm)):
            for f in fields(obj):
                env = os.environ.get(prefix + f.name.upper())
                if env is not None:
                    setattr(obj, f.name, _coerce(env, getattr(obj, f.name)))


def _apply_dc(obj: Any, data: Dict[str, Any]) -> None:
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        if value.lower() in ("", "none", "null"):
            return None
        try:
            return int(value)
        except ValueError:
            return value
    return value
