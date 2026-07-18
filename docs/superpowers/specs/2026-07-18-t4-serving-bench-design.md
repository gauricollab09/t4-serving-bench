# t4-serving-bench — Design

**Date:** 2026-07-18
**One-line:** Measure what an image token costs vs a text token when serving
Qwen2-VL-2B-Instruct with vLLM 0.25.1 on a free Colab T4, and whether prefix
caching helps repeated images.

## Purpose

Characterize the serving cost of a small vision-language model on commodity
hardware, with every number reproducible from committed code and results.
Deliverables: harness code, unit tests, results JSONL, plots, and a
report-style README.

**Headline result:** TTFT vs prompt-tokens for text-only and image inputs on
the *same model and server* (no cross-model confound). Slope ≈ per-token
prefill cost; the image curve's extra intercept ≈ fixed vision-encoder
overhead. Plus a measured cold-vs-warm answer for repeated-image prefix
caching (APC).

## Constraints

- Free Colab T4 only: fp16 (no bf16), 16 GB VRAM, session limits.
- Primary model Qwen2-VL-2B-Instruct; fp16 instability is the known top risk
  → EXP-0 correctness gate is mandatory; fallback is Phi-3.5-vision-instruct.
- `vllm==0.25.1` pinned; all multimodal flag syntax re-verified on that build.
- All harness code unit-testable on macOS CPU (no GPU, no vLLM import needed
  locally). Colab is a thin runner that clones this repo.

## Architecture

```
t4-serving-bench/
  bench/
    workloads.py   # text prompts at exact token counts (HF tokenizer);
                   # PIL noise images at 28-multiple sizes; base64 helper
    client.py      # OpenAI-compatible SSE streaming client: TTFT (first
                   # content delta), E2E, trailing usage.prompt_tokens
    server.py      # vLLM subprocess launcher: T4 flags, APC on/off configs,
                   # Phi fallback config, readiness poll, log capture
    runner.py      # experiment definitions; one JSON line per request
                   # appended to results/ (disconnect-safe, resumable);
                   # records vllm/transformers versions + vision-backend log
    analyze.py     # overlay plot, cold/warm comparison, p50/p95/p99 tables
  tests/           # CPU-only: token math, image sizing, SSE parse (canned)
  notebooks/colab_runner.ipynb   # clone repo → pip install → run experiments
  results/         # JSONL per run (committed)
  docs/superpowers/specs/        # this file
```

Interfaces: `workloads` produces `(payload_dict, expected_tokens, meta)`;
`client.send()` returns `{ttft_s, e2e_s, prompt_tokens, output_text}`;
`runner` composes the two and never imports vLLM (server is a subprocess).

## Model & launch (from verified guide)

```
vllm serve Qwen/Qwen2-VL-2B-Instruct --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 1003520}' --port 8000
```

Token formula (Qwen2-VL): after resize to multiples of 28,
`image_tokens = (H/28)·(W/28)`, +2 vision delimiters. Sweep squares
{224, 448, 672, 896}px → {64, 256, 576, 1024} image tokens.
`--limit-mm-per-prompt` JSON vs dotted syntax: verify on 0.25.1 at EXP-0.

## Experiments

- **EXP-0 — fp16 gate (go/no-go).** One 448×448 image, temperature=0,
  "Describe this image." Pass: coherent caption AND prompt_tokens ≈ 280.
  Fail (`!!!!`, empty, NaN, device-side assert): switch to Phi-3.5-vision
  config, rerun gate, and token accounting becomes empirical-only.
- **EXP-T — text TTFT curve.** Unique prompts (defeat APC) at {64, 256, 576,
  1024} prompt tokens, ~10 cold requests each, fixed small max_tokens.
- **EXP-V — image TTFT curve.** Unique noise images at the four sizes,
  ~10 each; assert measured prompt_tokens matches formula (+scaffolding).
- **EXP-C — repeated-image caching.** One fixed image, ~20 distinct
  questions, APC off vs on (server restart between; keep mm-preprocessor
  cache enabled). Compare cold vs warm TTFT; confirm cache hits in logs.
- **EXP-3 (optional)** — image vs token-matched text description of the same
  content.

LMCache: out of scope (confound); noted in README as future work.

## Work plan

1. Scaffold; `workloads.py` + `client.py` with tests green locally;
   Colab: install vllm 0.25.1, launch server, run EXP-0.
2. `server.py` + `runner.py`; run EXP-T + EXP-V; first overlay plot.
3. EXP-C runs + analysis.
4. README report, plots, publish.

## Risks & handling

| Risk | Handling |
|---|---|
| fp16 gibberish/NaN/CUDA assert (Qwen2-VL) | EXP-0 gate; Phi-3.5 fallback config ready in server.py |
| Colab disconnect mid-run | JSONL append per request; runner resumes by counting existing lines |
| OOM on 16 GB | mm pixel cap + image limit; fallback knobs: gpu-mem-util 0.8, max-model-len 2048 |
| Flag syntax drift on 0.25.1 | EXP-0 verifies launch flags; run metadata records exact versions |
| APC contaminating "cold" curves | unique prompt text / unique noise images per request |

## Testing

CPU-local unit tests before any GPU time (token counts, image dims, SSE
parsing on a canned stream). EXP-0 is the integration gate. Every number in
the README traces to a committed results JSONL.

## Success criteria

- Overlay plot with interpretable slope/intercept from ≥10 samples/point.
- Measured APC speedup (or measured null result) for repeated images.
- All experiments reproducible from the notebook in ≤1 Colab session.
