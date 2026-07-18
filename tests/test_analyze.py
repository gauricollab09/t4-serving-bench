from bench.analyze import fit_line, load_jsonl, median_ttft_by
from bench.runner import append_jsonl


def synthetic(prompt_tokens, ttft_s):
    return {"prompt_tokens": prompt_tokens, "ttft_s": ttft_s}


def test_fit_recovers_known_line():
    # ttft_ms = 100 + 2 * tokens, exactly
    records = [synthetic(t, (100 + 2 * t) / 1000) for t in (64, 256, 576, 1024)]
    slope, intercept = fit_line(records)
    assert abs(slope - 2.0) < 1e-6
    assert abs(intercept - 100.0) < 1e-3


def test_median_groups():
    records = [
        {"side_px": 224, "ttft_s": 0.1},
        {"side_px": 224, "ttft_s": 0.3},
        {"side_px": 224, "ttft_s": 0.2},
        {"side_px": 448, "ttft_s": 1.0},
    ]
    medians = median_ttft_by(records, "side_px")
    assert medians == {224: 0.2, 448: 1.0}


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "r.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    assert [r["a"] for r in load_jsonl(path)] == [1, 2]


def test_first_occurrence_drops_reruns():
    records = [
        {"side_px": 224, "seed": 1, "ttft_s": 0.1},
        {"side_px": 224, "seed": 2, "ttft_s": 0.1},
        {"side_px": 224, "seed": 1, "ttft_s": 0.05},  # rerun: not cold
    ]
    from bench.analyze import first_occurrence
    kept = first_occurrence(records, ("side_px", "seed"))
    assert len(kept) == 2
    assert kept[0]["ttft_s"] == 0.1
