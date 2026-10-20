"""
Stage 1 — semantic candidate retrieval.

This module reuses the original ESCOX mechanism (sentence-transformer embeddings of
"label + alternative labels + description" texts compared by cosine similarity with
embeddings of text fragments) and extends it in three ways:

1. it returns *ranked candidates with scores and the supporting text fragment*
   (the original only returned IDs above a threshold);
2. text fragments carry character offsets, so evidence can be highlighted;
3. the fragment splitter uses word boundaries for "and"/"or" (the original
   ``re.split(r"...|and|or")`` also split inside words such as "information").

The original algorithm is still available unchanged through
``esco_skill_extractor.SkillExtractor`` and is used as Experiment A baseline.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import CACHE_DIR

log = logging.getLogger("escox3.retrieval")

# --------------------------------------------------------------------------- chunking
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+|\r+|\t+|•|·|▪|◦|•")
_CLAUSE_SPLIT = re.compile(r",|\(|\)|/|\s+-\s+|\b(?:and|or|as well as|including|such as|e\.g\.)\b", re.I)


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    kind: str  # "sentence" | "clause" | "document"


def _spans(text: str, pattern: re.Pattern, offset: int = 0):
    pos = 0
    for m in pattern.finditer(text):
        yield text[pos:m.start()], offset + pos
        pos = m.end()
    yield text[pos:], offset + pos


def chunk_text(text: str, min_chars: int = 4, max_chunks: int = 400) -> List[Chunk]:
    """Split text into sentence-level and clause-level fragments with character offsets."""
    chunks: List[Chunk] = []
    seen = set()

    def add(raw: str, start: int, kind: str):
        stripped = raw.strip(" \t\r\n-–—*>|\"'.,;:")
        if len(stripped) < min_chars or not re.search(r"[A-Za-zÀ-ÿ]", stripped):
            return
        s = start + raw.find(stripped)
        key = stripped.lower()
        if key in seen:
            return
        seen.add(key)
        chunks.append(Chunk(stripped, s, s + len(stripped), kind))

    for sent, s0 in _spans(text, _SENT_SPLIT):
        add(sent, s0, "sentence")
        clauses = list(_spans(sent, _CLAUSE_SPLIT, s0))
        if len(clauses) > 1:
            for cl, c0 in clauses:
                add(cl, c0, "clause")
        if len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks]


# --------------------------------------------------------------------------- embedding indices
@dataclass
class Candidate:
    id: str
    score: float
    chunk_index: int
    rank: int = 0
    sources: List[str] = field(default_factory=lambda: ["semantic"])


class EmbeddingIndex:
    """A matrix of L2-normalised reference embeddings with cosine top-k search."""

    def __init__(self, name: str, ids: Sequence[str], embeddings: np.ndarray):
        self.name = name
        self.ids = list(ids)
        self.emb = np.asarray(embeddings, dtype=np.float32)
        self.pos = {u: i for i, u in enumerate(self.ids)}

    def similarities(self, query_emb: np.ndarray) -> np.ndarray:
        return query_emb @ self.emb.T

    def search(self, query_emb: np.ndarray, threshold: float, top_k_per_query: int,
               max_candidates: int) -> List[Candidate]:
        """Aggregate top-k matches over all query fragments (max-pooling per reference concept)."""
        if query_emb.size == 0:
            return []
        sims = self.similarities(query_emb)
        k = min(top_k_per_query, sims.shape[1])
        best: Dict[int, Candidate] = {}
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for qi in range(sims.shape[0]):
            for ri in top[qi]:
                s = float(sims[qi, ri])
                if s < threshold:
                    continue
                if ri not in best or s > best[ri].score:
                    best[ri] = Candidate(self.ids[ri], s, qi)
        ranked = sorted(best.values(), key=lambda c: -c.score)[:max_candidates]
        for r, c in enumerate(ranked, 1):
            c.rank = r
        return ranked

    def score_of(self, query_emb: np.ndarray, ref_id: str) -> tuple[float, int]:
        i = self.pos[ref_id]
        col = query_emb @ self.emb[i]
        j = int(np.argmax(col))
        return float(col[j]), j


class Embedder:
    """Thin wrapper around a SentenceTransformer, shared with the original ESCOX extractor."""

    def __init__(self, model, model_name: str, device: str):
        self.model = model
        self.model_name = model_name
        self.device = device
        self._lock = threading.Lock()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            e = self.model.encode(texts, batch_size=batch_size, device=self.device,
                                  normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return e.astype(np.float32)

    @property
    def dim(self) -> int:
        f = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(f())

    def build_index(self, name: str, ids: List[str], texts: List[str], cache_dir: Path = CACHE_DIR) -> EmbeddingIndex:
        """Embed reference texts once and cache them on disk (key = model + content hash)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(("\x1f".join(ids) + "\x1e" + "\x1f".join(texts)).encode()).hexdigest()[:16]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_name)[-60:]
        path = cache_dir / f"{name}__{slug}__{h}.npy"
        if path.exists():
            emb = np.load(path)
            log.info("Loaded cached %s embeddings (%d) from %s", name, len(ids), path.name)
        else:
            log.info("Embedding %d %s reference texts with %s ...", len(texts), name, self.model_name)
            emb = self.encode(texts)
            np.save(path, emb)
        return EmbeddingIndex(name, ids, emb)
