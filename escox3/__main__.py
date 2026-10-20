"""
ESCOX 3.0 command-line interface.

    python -m escox3                       # start the web application (same as `serve`)
    python -m escox3 serve --port 8000 [--llm-backend none|llama_cpp|transformers|openai] [--llm-model-path X.gguf]
    python -m escox3 analyse --file job.pdf [--modules skills,occupations,sectors] [--mode hybrid]
    python -m escox3 build-reference       # rebuild data/reference from data/raw
    python -m escox3 import-green-occupations list.csv
    python -m escox3 import-nace nace_structure.csv
    python -m escox3 info                  # taxonomy counts, versions and data gaps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _common(p: argparse.ArgumentParser):
    p.add_argument("--config", help="TOML configuration file (see config/escox3.example.toml)")
    p.add_argument("--embedding-model", "-m", help="Sentence-transformer model (default all-MiniLM-L6-v2)")
    p.add_argument("--device", "-d", help="torch device (default: cuda if available else cpu)")
    p.add_argument("--mode", choices=["hybrid", "retrieval", "llm_only"])
    p.add_argument("--llm-backend", choices=["llama_cpp", "transformers", "openai", "none"])
    p.add_argument("--llm-model-path", help="Local GGUF file for the llama_cpp backend")
    p.add_argument("--llm-repo", help="Hugging Face repo id of the GGUF model (llama_cpp backend)")
    p.add_argument("--llm-file", help="GGUF filename inside --llm-repo")
    p.add_argument("--llm-model", help="Model id for the transformers / openai backends")
    p.add_argument("--llm-api-base", help="Base URL of an OpenAI-compatible server (openai backend)")
    p.add_argument("--n-threads", type=int, help="CPU threads for llama.cpp")
    p.add_argument("--n-gpu-layers", type=int, help="Layers offloaded to GPU (llama.cpp GPU build)")
    p.add_argument("--no-legacy", action="store_true",
                   help="Do not load the original ESCOX extractor (disables /extract-skills, /extract-occupations)")
    p.add_argument("--verbose", "-v", action="store_true")


def _config(a):
    from .config import Config

    cfg = Config.load(a.config)
    if a.embedding_model:
        cfg.embedding_model = a.embedding_model
    if a.device:
        cfg.device = a.device
    if a.mode:
        cfg.mode = a.mode
    if a.llm_backend:
        cfg.llm.backend = a.llm_backend
    if a.llm_model_path:
        cfg.llm.model_path = a.llm_model_path
    if a.llm_repo:
        cfg.llm.hf_repo_id = a.llm_repo
    if a.llm_file:
        cfg.llm.hf_filename = a.llm_file
    if a.llm_model:
        cfg.llm.transformers_model = a.llm_model
        cfg.llm.api_model = a.llm_model
    if a.llm_api_base:
        cfg.llm.api_base = a.llm_api_base
    if a.n_threads:
        cfg.llm.n_threads = a.n_threads
    if a.n_gpu_layers is not None:
        cfg.llm.n_gpu_layers = a.n_gpu_layers
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="escox3", description="ESCOX 3.0 - Green Skills, Occupations & Sector Intelligence")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="start the web application")
    _common(s)
    s.add_argument("--host", "-c", default=None)
    s.add_argument("--port", "-p", type=int, default=None)

    an = sub.add_parser("analyse", help="analyse a text or document and print JSON")
    _common(an)
    an.add_argument("--text")
    an.add_argument("--file")
    an.add_argument("--modules", default="skills,occupations,sectors")
    an.add_argument("--csv", help="also write CSV to this path")

    sub.add_parser("build-reference", help="(re)build the reference store from data/raw")
    g = sub.add_parser("import-green-occupations", help="import an authoritative green-occupation list")
    g.add_argument("csv")
    n = sub.add_parser("import-nace", help="import the official NACE Rev. 2.1 structure (code,title)")
    n.add_argument("csv")
    sub.add_parser("info", help="show taxonomy counts, versions and data gaps")

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        argv = ["serve"] + list(argv)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(a, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "build-reference":
        from .build_reference import build
        print(json.dumps(build()["counts"], indent=2))
        return
    if a.cmd == "import-green-occupations":
        from .build_reference import import_green_occupations, build
        print(f"Imported {import_green_occupations(a.csv)} rows"); build()
        return
    if a.cmd == "import-nace":
        from .build_reference import import_nace_structure
        print(f"Imported {import_nace_structure(a.csv)} NACE codes")
        return
    if a.cmd == "info":
        from .taxonomy import ReferenceStore
        print(json.dumps(ReferenceStore().summary(), indent=2, ensure_ascii=False))
        return

    cfg = _config(a)
    from .pipeline import ESCOX3

    if a.cmd == "analyse":
        from .export import to_csv
        from .text_extraction import extract_text

        if a.file:
            text, _ = extract_text(a.file, Path(a.file).read_bytes())
        elif a.text:
            text = a.text
        else:
            text = sys.stdin.read()
        mods = {m.strip() for m in a.modules.split(",")}
        engine = ESCOX3(cfg, load_legacy=not a.no_legacy)
        res = engine.analyse(text, "skills" in mods, "occupations" in mods, "sectors" in mods)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if a.csv:
            Path(a.csv).write_text(to_csv(res))
        return

    # serve
    from waitress import serve
    from .server import create_app

    if a.host:
        cfg.host = a.host
    if a.port:
        cfg.port = a.port
    engine = ESCOX3(cfg, load_legacy=not a.no_legacy)
    app = create_app(engine)
    print(f"\n  ESCOX 3.0 is running at  http://{cfg.host}:{cfg.port}\n")
    serve(app, host=cfg.host, port=cfg.port, channel_timeout=12000, threads=4)


if __name__ == "__main__":
    main()
