import numpy as np
import torch

from evaluation.codebook_usage import (
    _gini,
    analyze_codebook_usage,
    save_codebook_usage_report,
)


def test_gini_extremes():
    # Uniform -> 0; single-code collapse -> approaches 1 (for large K).
    assert _gini(np.ones(10)) == 0.0
    collapsed = np.zeros(10)
    collapsed[0] = 100.0
    assert 0.8 < _gini(collapsed) <= 1.0


def test_active_dead_counts():
    codes = torch.tensor([0, 0, 0, 1, 1, 2, 5, 5, 5], dtype=torch.int64)
    stats = analyze_codebook_usage(codes, num_codes=8, min_freq_frac=0.1)
    # Codes 3, 4, 6, 7 never used -> 4 dead.
    assert stats["dead_codes"] == 4
    assert stats["used_codes"] == 4
    # Active threshold 10%: codes 0 (3/9), 5 (3/9) are >= 0.1; code 2 (1/9) is ~0.11.
    assert stats["active_codes"] >= 2
    # Entropy / perplexity in valid ranges for K=8.
    assert 0.0 <= stats["entropy"] <= np.log(8) + 1e-6
    assert 1.0 <= stats["perplexity"] <= 8.0


def test_collapse_detected_on_skewed():
    # Almost all mass on one code.
    codes = torch.tensor([0] * 95 + [1, 2, 3, 4, 5], dtype=torch.int64)
    stats = analyze_codebook_usage(codes, num_codes=8)
    assert stats["collapse_score"] > 0.7
    assert stats["perplexity"] < 3.0


def test_empty_codes():
    stats = analyze_codebook_usage(torch.empty(0, dtype=torch.int64), num_codes=4)
    assert stats["active_codes"] == 0
    assert stats["dead_codes"] == 4
    assert stats["perplexity"] == 0.0


def test_report_writes_csv(tmp_path):
    codes = torch.randint(0, 6, (100,))
    stats = analyze_codebook_usage(codes, num_codes=6)
    csv = tmp_path / "usage.csv"
    png = tmp_path / "usage.png"
    save_codebook_usage_report(stats, str(csv), str(png))
    assert csv.exists()
    assert png.exists()
