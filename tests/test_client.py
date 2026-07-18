"""client.py is tested against a tiny fake server that speaks SSE the way
vLLM does, so the parsing and timing logic is verified without any GPU."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench.client import send_request

SSE_BODY = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":258,"completion_tokens":2}}\n\n'
    b'data: [DONE]\n\n'
)


class FakeVllm(BaseHTTPRequestHandler):
    def do_POST(self):
        # read the request body first; replying without reading it resets
        # the connection and the client sees a broken stream
        body_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(body_len)
        self.send_response(200)
        self.send_header("Content-Length", str(len(SSE_BODY)))
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(SSE_BODY)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeVllm)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_stream_is_parsed_and_timed(fake_server):
    result = send_request({"model": "m", "messages": []}, base_url=fake_server)
    assert result["output_text"] == "Hello world"
    assert result["prompt_tokens"] == 258
    assert result["completion_tokens"] == 2
    assert result["ttft_s"] is not None
    assert result["ttft_s"] <= result["e2e_s"]
