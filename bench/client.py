"""The stopwatch: send one streaming request, time it, read the token bill.

The server streams its reply as server-sent events (SSE): one
"data: {json}" line per chunk, arriving as the model generates. Reading
the stream chunk-by-chunk is what makes TTFT measurable at all -- a
non-streaming reply would only reveal when the WHOLE answer finished.
"""

import json
import time

import requests


def send_request(payload: dict, base_url: str = "http://localhost:8000") -> dict:
    """POST one chat completion and stream the reply.

    Returns ttft_s (send -> first ANSWER TEXT), e2e_s (send -> stream end),
    the server-reported token counts, and the full output text.
    """
    # stream=True: reply arrives chunk-by-chunk (enables the stopwatch).
    # include_usage: the final chunk carries usage.prompt_tokens -- the
    # server's own bill for what we sent, image tokens included. All token
    # counts in our data come from this receipt, never our own estimate.
    payload = dict(payload, stream=True, stream_options={"include_usage": True})

    t_start = time.perf_counter()
    ttft = None
    parts = []
    usage = None

    url = f"{base_url}/v1/chat/completions"
    with requests.post(url, json=payload, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            # SSE frames look like:  data: {...json...}
            # blank keep-alive lines and anything else are skipped
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):]
            if data == b"[DONE]":  # server's end-of-stream sentinel
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                # final chunk: empty choices + the usage "receipt"
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                # IMPORTANT: the first chunk is often housekeeping (a delta
                # carrying only role="assistant", no text). The clock stops
                # at the first NON-EMPTY content -- the first visible word.
                # Stopping at "first chunk of any kind" would measure TTFB,
                # not TTFT, and read one chunk too early.
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    parts.append(content)

    return {
        "ttft_s": ttft,
        "e2e_s": time.perf_counter() - t_start,
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "completion_tokens": usage["completion_tokens"] if usage else None,
        "output_text": "".join(parts),
    }
