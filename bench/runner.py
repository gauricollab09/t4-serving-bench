"""Experiment runners: fire calibrated requests, append one JSON line each.

Appending line-by-line means a dropped Colab session loses nothing already
measured. Every record carries the server-reported prompt_tokens, which is
the x value used in analysis (never our own estimate).
"""

import json

from bench.client import send_request
from bench.workloads import (
    image_payload,
    noise_image,
    text_payload,
    text_prompt,
    to_data_url,
)


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def warmup(model: str, tokenizer, base_url: str = "http://localhost:8000") -> None:
    """Throwaway requests so one-time costs are paid before any measured one.

    The first request after server start triggers Triton kernel
    compilation (~45x slower than steady state). One warmup per image
    size, since kernels are shape-specific; results never recorded."""
    print("warmup: text")
    send_request(text_payload(model, text_prompt(64, tokenizer, seed=999_000), max_tokens=8), base_url)
    for side in (224, 448, 672, 896):
        print(f"warmup: image {side}px")
        payload = image_payload(model, to_data_url(noise_image(side, seed=999_000 + side)), "Describe.", max_tokens=8)
        send_request(payload, base_url)


def run_text_sweep(model, tokenizer, targets, runs_per_target, out_path,
                   base_url: str = "http://localhost:8000") -> None:
    """EXP-T: unique text prompts at each target token count.

    seed increments per request -> every prompt is unique -> the prefix
    cache can never warm a "cold" measurement. ~10 runs per point so
    medians are robust to free-tier GPU noise."""
    seed = 0
    for target in targets:
        for run in range(runs_per_target):
            seed += 1
            prompt = text_prompt(target, tokenizer, seed)
            result = send_request(text_payload(model, prompt, max_tokens=16), base_url)
            append_jsonl(out_path, {
                "experiment": "EXP-T",
                "target_tokens": target,
                "seed": seed,
                "prompt_tokens": result["prompt_tokens"],
                "ttft_s": result["ttft_s"],
                "e2e_s": result["e2e_s"],
            })
            print(f"EXP-T {target:>4} tok  run {run + 1}/{runs_per_target}  "
                  f"prompt_tokens={result['prompt_tokens']}  ttft={result['ttft_s']:.3f}s")


def run_image_sweep(model, sides, runs_per_side, out_path,
                    base_url: str = "http://localhost:8000") -> None:
    """EXP-V: unique noise images at each resolution.

    Same discipline as the text sweep: unique image per request (seed),
    fixed question, small max_tokens (TTFT doesn't depend on it; short
    decode just makes the sweep faster)."""
    seed = 0
    for side in sides:
        for run in range(runs_per_side):
            seed += 1
            data_url = to_data_url(noise_image(side, seed))
            payload = image_payload(model, data_url, "Describe this image.", max_tokens=16)
            result = send_request(payload, base_url)
            append_jsonl(out_path, {
                "experiment": "EXP-V",
                "side_px": side,
                "seed": seed,
                "prompt_tokens": result["prompt_tokens"],
                "ttft_s": result["ttft_s"],
                "e2e_s": result["e2e_s"],
            })
            print(f"EXP-V {side:>3}px  run {run + 1}/{runs_per_side}  "
                  f"prompt_tokens={result['prompt_tokens']}  ttft={result['ttft_s']:.3f}s")


def run_repeated_image(model, side, image_seed, n_questions, out_path, config,
                       image_first, base_url: str = "http://localhost:8000") -> None:
    """EXP-C: one fixed image, many distinct questions. Request 1 is always
    cold; whether 2..n go faster depends on the cache config and on whether
    the image precedes the changing question in the prompt."""
    data_url = to_data_url(noise_image(side, image_seed))
    for i in range(1, n_questions + 1):
        question = f"Question {i}: describe aspect number {i} of this image."
        payload = image_payload(model, data_url, question, max_tokens=16,
                                image_first=image_first)
        result = send_request(payload, base_url)
        append_jsonl(out_path, {
            "experiment": "EXP-C",
            "config": config,
            "request_index": i,
            "side_px": side,
            "prompt_tokens": result["prompt_tokens"],
            "ttft_s": result["ttft_s"],
            "e2e_s": result["e2e_s"],
        })
        print(f"EXP-C [{config}] req {i:>2}/{n_questions}  "
              f"prompt_tokens={result['prompt_tokens']}  ttft={result['ttft_s']:.3f}s")
