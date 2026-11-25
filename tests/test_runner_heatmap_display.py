"""E1-1 heatmap aggregation / display defaults of the runners (CPU-only, no torch / matplotlib / ComfyUI).

Defaults follow ``reference_repo/Anima_HEATMAP/anima_heatmap/maps.py`` v1: span ``mean`` (L117),
``render_map(colormap="turbo", normalization="relative")`` (L145-170). Legacy behavior stays
available as ``--span-agg sum`` / ``--heatmap-norm shared`` / ``--heatmap-cmap inferno``.
"""

from __future__ import annotations

import argparse

import pytest

np = pytest.importorskip("numpy")

from xai_dit.runners import e1_e4_profile as prof  # noqa: E402  (torch-free at import time)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    prof.add_heatmap_display_args(p)
    return p


def test_defaults_match_anima_heatmap_v1():
    assert (prof.DEFAULT_SPAN_AGG, prof.DEFAULT_HEATMAP_NORM, prof.DEFAULT_HEATMAP_CMAP) == ("mean", "per_tag", "turbo")
    args = _parser().parse_args([])
    assert (args.heatmap_norm, args.heatmap_cmap) == ("per_tag", "turbo")


def test_profile_cli_defaults_and_legacy_flags():
    """``e1_e4_profile.main`` parser: defaults and opt-in legacy values (parsed only; no run)."""
    p = argparse.ArgumentParser()
    prof.add_heatmap_display_args(p)
    p.add_argument("--span-agg", choices=prof.SPAN_AGG_CHOICES, default=prof.DEFAULT_SPAN_AGG)
    assert p.parse_args([]).span_agg == "mean"
    legacy = p.parse_args(["--span-agg", "sum", "--heatmap-norm", "shared", "--heatmap-cmap", "inferno"])
    assert (legacy.span_agg, legacy.heatmap_norm, legacy.heatmap_cmap) == ("sum", "shared", "inferno")
    with pytest.raises(SystemExit):
        p.parse_args(["--heatmap-cmap", "viridis"])


def test_span_mean_removes_token_count_scaling():
    """Same per-token attention on a 10-token and a 2-token span: mean is equal, sum scales 5x."""
    per_query = np.full((4, 10), 0.01)  # [n_img, span tokens]
    long_sum, short_sum = per_query.sum(-1), per_query[:, :2].sum(-1)
    long_mean = long_sum / prof.span_agg_divisor(10, "mean")
    short_mean = short_sum / prof.span_agg_divisor(2, "mean")
    np.testing.assert_allclose(long_mean, short_mean)
    np.testing.assert_allclose(long_mean, per_query.mean(-1))
    assert prof.span_agg_divisor(10, "sum") == 1.0
    np.testing.assert_allclose(long_sum / short_sum, 5.0)


def test_span_mean_exact_across_key_chunks():
    """The runner divides each key-chunk partial sum by the FULL span length -> exact mean."""
    row = np.arange(6, dtype=np.float64)[None, :]  # one query, 6-token span split 4 + 2 by key chunks
    div = prof.span_agg_divisor(6, "mean")
    chunked = row[:, :4].sum(-1) / div + row[:, 4:].sum(-1) / div
    np.testing.assert_allclose(chunked, row.mean(-1))


def test_span_agg_rejects_unknown():
    with pytest.raises(ValueError):
        prof.span_agg_divisor(3, "max")
    with pytest.raises(ValueError):
        prof.heatmap_range([np.zeros(2)], "absolute")


def test_per_tag_norm_is_relative_min_max():
    weak = np.linspace(0.001, 0.002, 64).reshape(8, 8)  # short-tag map, far below a strong tag
    lo, hi = prof.heatmap_range([weak], "per_tag")
    assert (lo, hi) == pytest.approx((0.001, 0.002))
    n = prof.normalize_heatmap(weak, lo, hi)
    assert float(n.min()) == pytest.approx(0.0, abs=1e-6) and float(n.max()) == pytest.approx(1.0, rel=1e-6)


def test_shared_norm_reproduces_legacy_vmax_clip():
    """``shared`` = old behavior: one 0..99.5th-percentile scale; the weak tag is crushed by the strong one."""
    rng = np.random.default_rng(0)
    strong, weak = rng.uniform(0, 1.0, (8, 8)), rng.uniform(0, 0.05, (8, 8))
    lo, hi = prof.heatmap_range([strong, weak], "shared")
    legacy_vmax = float(np.percentile(np.concatenate([strong.ravel(), weak.ravel()]), 99.5))
    assert lo == 0.0 and hi == pytest.approx(legacy_vmax)
    np.testing.assert_allclose(prof.normalize_heatmap(weak, lo, hi),
                               np.clip(weak / max(legacy_vmax, 1e-12), 0, 1), rtol=1e-6)
    assert float(prof.normalize_heatmap(weak, lo, hi).max()) < 0.1  # why per_tag is the default
    assert float(prof.normalize_heatmap(weak, *prof.heatmap_range([weak], "per_tag")).max()) == pytest.approx(1.0)


def test_constant_map_does_not_divide_by_zero():
    flat = np.full((4, 4), 0.3)
    n = prof.normalize_heatmap(flat, *prof.heatmap_range([flat], "per_tag"))
    assert np.isfinite(n).all() and float(n.max()) == 0.0
    sm = prof.prepare_heatmap(flat, "per_tag", smooth_sigma=1.5, clip_pct=99.0)
    assert np.isfinite(sm).all() and float(sm.max()) == pytest.approx(0.0, abs=1e-6)


# --- display smoothing + per_tag upper-percentile clipping (pure NumPy) ---

def test_smooth_and_clip_cli_defaults_and_validation():
    args = _parser().parse_args([])
    assert (args.heatmap_smooth_sigma, args.heatmap_clip_pct) == (1.5, 99.0)
    assert (prof.DEFAULT_HEATMAP_SMOOTH_SIGMA, prof.DEFAULT_HEATMAP_CLIP_PCT) == (1.5, 99.0)
    off = _parser().parse_args(["--heatmap-smooth-sigma", "0", "--heatmap-clip-pct", "100"])
    assert (off.heatmap_smooth_sigma, off.heatmap_clip_pct) == (0.0, 100.0)
    for bad in (["--heatmap-smooth-sigma", "-1"], ["--heatmap-clip-pct", "0"], ["--heatmap-clip-pct", "100.5"]):
        with pytest.raises(SystemExit):
            _parser().parse_args(bad)


def test_gaussian_kernel_matches_closed_form():
    k = prof.gaussian_kernel_1d(1.5)
    assert k.size == 2 * int(4.0 * 1.5 + 0.5) + 1 and k.sum() == pytest.approx(1.0)
    x = np.arange(k.size) - k.size // 2
    ref = np.exp(-0.5 * (x / 1.5) ** 2)
    np.testing.assert_allclose(k, ref / ref.sum())
    np.testing.assert_allclose(k, k[::-1])


def test_smooth_is_separable_gaussian_and_mass_preserving():
    sigma = 1.5
    delta = np.zeros((32, 32))
    delta[16, 16] = 1.0
    out = prof.smooth_heatmap(delta, sigma)
    k = prof.gaussian_kernel_1d(sigma)
    r = k.size // 2
    # interior impulse response == outer product of the 1D kernel (2D Gaussian)
    np.testing.assert_allclose(out[16 - r:16 + r + 1, 16 - r:16 + r + 1], np.outer(k, k), atol=1e-12)
    assert out.sum() == pytest.approx(1.0)
    assert int(np.argmax(out)) == 16 * 32 + 16
    # symmetric (reflect) borders: a corner spike keeps its mass instead of leaking into zero padding
    corner = np.zeros((32, 32))
    corner[0, 0] = 1.0
    assert prof.smooth_heatmap(corner, sigma).sum() == pytest.approx(1.0)
    # constant map is a fixed point; sigma=0 is the identity
    np.testing.assert_allclose(prof.smooth_heatmap(np.full((8, 8), 0.3), sigma), 0.3)
    rng = np.random.default_rng(1)
    m = rng.uniform(size=(8, 8))
    np.testing.assert_array_equal(prof.smooth_heatmap(m, 0.0), m)
    with pytest.raises(ValueError):
        prof.smooth_heatmap(np.zeros((2, 2, 2)), 1.0)


def test_smooth_handles_maps_smaller_than_kernel():
    m = np.arange(6, dtype=np.float64).reshape(2, 3)  # kernel radius 6 > both dims
    out = prof.smooth_heatmap(m, 1.5)
    assert out.shape == m.shape and np.isfinite(out).all()
    assert m.min() <= out.min() and out.max() <= m.max()


def test_per_tag_clip_pct_range():
    rng = np.random.default_rng(2)
    m = rng.uniform(0.0, 1.0, (64, 64))
    assert prof.heatmap_range([m], "per_tag") == prof.heatmap_range([m], "per_tag", 100.0)
    assert prof.heatmap_range([m], "per_tag", 100.0) == (pytest.approx(m.min()), pytest.approx(m.max()))
    lo, hi = prof.heatmap_range([m], "per_tag", 99.0)
    assert lo == pytest.approx(m.min()) and hi == pytest.approx(np.percentile(m, 99.0))
    # shared ignores clip_pct (legacy 0..99.5%)
    assert prof.heatmap_range([m], "shared", 50.0) == prof.heatmap_range([m], "shared")
    for bad in (0.0, -1.0, 101.0):
        with pytest.raises(ValueError):
            prof.heatmap_range([m], "per_tag", bad)


def _spiky_map() -> "np.ndarray":
    """Smooth blob (peak ~1) plus a single corner patch 4.65x above the blob peak."""
    yy, xx = np.mgrid[0:64, 0:64]
    blob = np.exp(-((yy - 40) ** 2 + (xx - 24) ** 2) / (2 * 8.0 ** 2))
    m = 0.05 + blob
    m[0, 0] = 4.65 * float(m.max())
    return m


def test_corner_spike_crushes_exact_min_max_but_not_p99():
    m = _spiky_map()
    blob_peak = (40, 24)
    exact = prof.normalize_heatmap(m, *prof.heatmap_range([m], "per_tag", 100.0))
    assert float(exact[blob_peak]) < 0.25  # the real signal is crushed into the bottom of the colormap
    p99 = prof.normalize_heatmap(m, *prof.heatmap_range([m], "per_tag", 99.0))
    assert float(p99[blob_peak]) == pytest.approx(1.0)  # saturates at the top
    assert float(p99[0, 0]) == 1.0 and float(p99.min()) == pytest.approx(0.0, abs=1e-6)


def test_prepare_heatmap_smooths_before_clipping():
    m = _spiky_map()
    out = prof.prepare_heatmap(m, "per_tag", smooth_sigma=1.5, clip_pct=99.0)
    assert out.dtype == np.float32 and out.shape == m.shape
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    # equals the explicit pipeline: smooth -> range(p99 of smoothed) -> normalize
    s = prof.smooth_heatmap(m, 1.5)
    np.testing.assert_allclose(out, prof.normalize_heatmap(s, *prof.heatmap_range([s], "per_tag", 99.0)))
    # the smoothed spike's own percentile, not the raw map's, sets the scale
    raw_scale = prof.normalize_heatmap(s, *prof.heatmap_range([m], "per_tag", 99.0))
    assert not np.allclose(out, raw_scale)
    assert float(out[40, 24]) > 0.9  # blob stays near the top of the colormap


def test_prepare_heatmap_defaults_are_backward_compatible():
    rng = np.random.default_rng(3)
    m = rng.uniform(0.001, 0.002, (8, 8))
    np.testing.assert_array_equal(prof.prepare_heatmap(m),
                                  prof.normalize_heatmap(m, *prof.heatmap_range([m], "per_tag")))
    # explicit group range (shared) bypasses per-map range; resize hook runs after smoothing
    np.testing.assert_allclose(prof.prepare_heatmap(m, rng=(0.0, 0.004)), np.clip(m / 0.004, 0, 1), rtol=1e-6)
    up = prof.prepare_heatmap(m, smooth_sigma=1.0, resize=lambda a: np.kron(a, np.ones((2, 2))))
    assert up.shape == (16, 16)


def test_scale_label_mentions_clip_and_sigma():
    assert prof.heatmap_scale_label("per_tag", 1.5, 99.0) == "per-panel min–p99 scale, gauss σ=1.5px"
    assert prof.heatmap_scale_label("per_tag", 0.0, 100.0) == "per-panel min–max scale, no smoothing"
    assert "99.5%" in prof.heatmap_scale_label("shared", 1.5, 99.0)
