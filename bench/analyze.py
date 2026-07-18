"""Turn results JSONL into line fits and summary tables.

The model: TTFT = intercept + slope * prompt_tokens.
Slope = per-token prefill cost. Comparing the text fit against the image
fit answers the headline question; the image intercept minus the text
intercept estimates the fixed vision-encoder overhead.
"""

import json

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_line(records: list[dict]) -> tuple[float, float]:
    """Least-squares TTFT vs prompt_tokens.
    Returns (slope in ms per token, intercept in ms).

    x is always the SERVER-reported prompt_tokens from each record, never
    our own estimate. Caveat: the fit is only meaningful if x actually
    varies -- Phi bills identical tokens for every image size, making its
    image "slope" degenerate (numpy warns; don't quote it)."""
    x = np.array([r["prompt_tokens"] for r in records], dtype=float)
    y = np.array([r["ttft_s"] for r in records], dtype=float) * 1000.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def median_ttft_by(records: list[dict], key: str) -> dict:
    """Median TTFT (seconds) per group, e.g. per target_tokens or side_px.

    Median, not mean: free-tier T4s produce occasional outliers (e.g. a
    1.1s spike in a 0.65s cluster); the median ignores them."""
    groups: dict = {}
    for r in records:
        groups.setdefault(r[key], []).append(r["ttft_s"])
    return {k: float(np.median(v)) for k, v in sorted(groups.items())}


def first_occurrence(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Keep only the first record per key combination. Guards analysis
    against a sweep cell accidentally run twice: repeated inputs are no
    longer cold, so only the first pass counts as a cold measurement."""
    seen = set()
    kept = []
    for r in records:
        key = tuple(r.get(k) for k in keys)
        if key not in seen:
            seen.add(key)
            kept.append(r)
    return kept
