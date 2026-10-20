"""OpenAI-compatible backend against a local stub server (checks request format and schema fallback)."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escox3.config import LLMConfig  # noqa: E402
from escox3.llm.backends import OpenAICompatibleBackend  # noqa: E402

SEEN = []


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            self.send_response(400); self.end_headers(); self.wfile.write(b"{}"); return
        out = {"choices": [{"message": {"content": '{"selected": []}'}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, *a):
        pass


def test_schema_fallback():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = LLMConfig(backend="openai", api_base=f"http://127.0.0.1:{srv.server_port}/v1", api_model="stub")
    b = OpenAICompatibleBackend(cfg)
    r = b.generate([{"role": "user", "content": "hi"}], schema={"type": "object"})
    srv.shutdown()
    assert r.text == '{"selected": []}' and r.prompt_tokens == 5 and r.error is None
    assert SEEN[0]["response_format"]["type"] == "json_schema" and SEEN[1]["response_format"]["type"] == "json_object"
    assert SEEN[0]["temperature"] == 0.0 and SEEN[0]["seed"] == 42
