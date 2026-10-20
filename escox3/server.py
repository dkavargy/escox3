"""
ESCOX 3.0 web application (Flask, served by waitress like the original ESCOX).

Endpoints
---------
GET  /                       one-page GUI
POST /api/analyse            analyse text (JSON) -> structured result
POST /api/extract-text       upload TXT/PDF/DOCX (multipart 'file') -> extracted text
POST /api/export/csv         result JSON -> CSV download
POST /api/export/json        result JSON -> JSON download
GET  /api/health             models / LLM status
GET  /api/taxonomy           reference-data summary, versions and data gaps
GET  /api/resolve?id=...     identifier -> preferred label (mapping layer)
POST /extract-skills         ORIGINAL ESCOX endpoint (unchanged behaviour)
POST /extract-occupations    ORIGINAL ESCOX endpoint (unchanged behaviour)
"""

from __future__ import annotations

import logging
import time
import traceback

from flask import Flask, Response, jsonify, render_template, request

from . import __version__
from .export import to_csv, to_json
from .text_extraction import ExtractionError, extract_text

log = logging.getLogger("escox3.server")


def _bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def create_app(engine) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = engine.cfg.max_upload_mb * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"File too large (limit {engine.cfg.max_upload_mb} MB)."), 413

    @app.route("/")
    def index():
        llm = engine.llm.info()
        return render_template("index.html", version=__version__, llm_available=llm.get("available"),
                               llm_name=llm.get("model_name") or "", languages=engine.store.nace_languages,
                               default_mode=engine.cfg.mode)

    @app.route("/api/analyse", methods=["POST", "OPTIONS"])
    def analyse():
        if request.method == "OPTIONS":
            return ("", 204)
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="Send a JSON object with at least a 'text' field."), 400
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify(error="Please provide some text to analyse."), 400
        flags = dict(extract_green_skills=_bool(body.get("extract_green_skills"), True),
                     extract_green_occupations=_bool(body.get("extract_green_occupations"), True),
                     extract_sectors=_bool(body.get("extract_sectors"), True))
        if not any(flags.values()):
            return jsonify(error="Select at least one extraction module."), 400
        try:
            result = engine.analyse(text, **flags, mode=body.get("mode"), label_language=body.get("label_language"))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except Exception as e:  # pragma: no cover
            log.error("Analysis failed: %s", traceback.format_exc())
            return jsonify(error=f"Internal error during analysis: {type(e).__name__}: {e}"), 500
        return Response(to_json(result), mimetype="application/json")

    @app.route("/api/extract-text", methods=["POST"])
    def extract():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify(error="No file received."), 400
        t0 = time.time()
        try:
            text, info = extract_text(f.filename, f.read())
        except ExtractionError as e:
            return jsonify(error=str(e)), 422
        except Exception as e:  # pragma: no cover
            log.error("Extraction failed: %s", traceback.format_exc())
            return jsonify(error=f"Text extraction failed ({type(e).__name__})."), 500
        info["seconds"] = round(time.time() - t0, 3)
        return jsonify(text=text, info=info)

    @app.route("/api/export/csv", methods=["POST"])
    def export_csv():
        result = request.get_json(silent=True)
        if not isinstance(result, dict):
            return jsonify(error="Send an ESCOX result object."), 400
        return Response(to_csv(result), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=escox3_results.csv"})

    @app.route("/api/export/json", methods=["POST"])
    def export_json():
        result = request.get_json(silent=True)
        if not isinstance(result, dict):
            return jsonify(error="Send an ESCOX result object."), 400
        return Response(to_json(result), mimetype="application/json",
                        headers={"Content-Disposition": "attachment; filename=escox3_results.json"})

    @app.route("/api/health")
    def health():
        return jsonify(status="ok", version=__version__, mode=engine.cfg.mode,
                       embedding_model=engine.cfg.embedding_model, device=engine.device,
                       legacy_escox_loaded=engine.legacy is not None, llm=engine.llm.info())

    @app.route("/api/taxonomy")
    def taxonomy():
        return jsonify(engine.store.summary())

    @app.route("/api/resolve")
    def resolve():
        ident = request.args.get("id", "")
        return jsonify(engine.store.resolve_label(ident, request.args.get("lang", "en")))

    # ---------------- original ESCOX API (kept for backward compatibility)
    @app.route("/extract-skills", methods=["POST"])
    def legacy_skills():
        return jsonify(engine.legacy_skills(request.json))

    @app.route("/extract-occupations", methods=["POST"])
    def legacy_occupations():
        return jsonify(engine.legacy_occupations(request.json))

    return app
