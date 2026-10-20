"""ESCOX 3.0 — Green Skills, Occupations & Sector Intelligence.

Extends the original ESCOX (esco_skill_extractor) with green skill extraction, green
occupation identification (ISCO-08) and economic sector classification (NACE Rev. 2.1)
using hybrid semantic retrieval + LLM validation.
"""

__version__ = "3.0.0"


def __getattr__(name):  # lazy import keeps `import escox3` light
    if name == "ESCOX3":
        from .pipeline import ESCOX3
        return ESCOX3
    if name == "Config":
        from .config import Config
        return Config
    raise AttributeError(name)
