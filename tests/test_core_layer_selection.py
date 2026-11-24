"""Direction 02 layer-selection invariants (``plan/m1.md`` §3 E2; ``docs/specs/02_layer_selection.md``).

E2-1 True Block Bypass (inner-call count == 0), cosine vitality, elbow / top-20% selection,
split-half Spearman, paired bootstrap; E2-2 ``Cont^(l)`` through a ``wo(x)`` *module call*
(quantization-safe), ``raw_mvo`` vs ``gated_mvo``, Top/Bottom/Random sets.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed; run on a torch host")

from xai_dit.adapters.base import COND, UNCOND, InterventionSpec, StepWindow, TokenSpan  # noqa: E402
from xai_dit.core.layer_selection import (  # noqa: E402
    ContributionTracker,
    attention_contribution,
    contribution_from_context,
    cosine_vitality,
    paired_bootstrap_ci,
    relative_depth_blocks,
    select_elbow,
    select_top_fraction,
    select_vital_layers,
    spearman_rho,
    split_half_spearman,
    top_bottom_random,
)
from xai_dit.core.session import XAISession  # noqa: E402


def maxdiff(a, b) -> float:
    return (a.double() - b.double()).abs().max().item()


# =========================================================================== E2-1 True Block Bypass


class _CountingInner(torch.nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.lin(x)


class _Block(torch.nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.attn, self.mlp = _CountingInner(d), _CountingInner(d)

    def forward(self, x):
        h = x + self.attn(x)
        return h + self.mlp(h)


class _DummyDiT(torch.nn.Module):
    """Adapter-style forward: consults ``session.bypass_mode`` before entering a block."""

    def __init__(self, n_blocks: int = 4, d: int = 16) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block(d) for _ in range(n_blocks))

    def forward(self, x, session=None, ctx=None):
        for i, blk in enumerate(self.blocks):
            if session is not None and session.bypass_mode(i, ctx) == "full_block":
                continue  # true bypass: identity residual, block never entered
            x = blk(x)
        return x

    def calls(self):
        return [(b.attn.calls, b.mlp.calls) for b in self.blocks]


def test_true_block_bypass_inner_call_count_is_zero(capabilities, make_ctx):
    torch.manual_seed(0)
    model = _DummyDiT()
    x = torch.randn(2, 24, 16)  # cond + uncond rows: bypass applies to both identically
    ctx = make_ctx((COND, UNCOND), chunk=1)
    sess = XAISession(capabilities["moss"], interventions=[InterventionSpec("true_block_bypass", blocks=[2])])
    with torch.no_grad():
        out = model(x, sess, ctx)
        assert model.calls() == [(1, 1), (1, 1), (0, 0), (1, 1)]
        ref = model.blocks[3](model.blocks[1](model.blocks[0](x)))
    assert maxdiff(out, ref) == 0
    assert sess.bypass_mode(2, ctx) == "full_block"
    assert sess.bypass_mode(2, make_ctx((UNCOND,), chunk=2)) == "full_block"
    assert sess.bypass_mode(1, ctx) is None


def test_true_block_bypass_window_and_mode(capabilities, make_ctx):
    spec = InterventionSpec("true_block_bypass", blocks=[0], window=StepWindow(end=1), params={"mode": "attn_only"})
    sess = XAISession(capabilities["anima"], interventions=[spec])
    assert sess.bypass_mode(0, make_ctx(sigma=1.0)) == "attn_only"
    assert sess.bypass_mode(0, make_ctx(sigma=0.875)) is None
    with sess.disabled():
        assert sess.bypass_mode(0, make_ctx(sigma=1.0)) is None


# =========================================================================== E2-1 vitality & selection


def test_cosine_vitality_range():
    full = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    orth = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    vit = cosine_vitality(full, torch.stack([full, orth, -full]))
    assert vit.shape == (3, 2)
    assert maxdiff(vit, torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])) < 1e-12


ELBOW = torch.tensor([1.0, 0.95, 0.9, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02])


def test_select_vital_layers_elbow_first():
    perm = torch.tensor([4, 0, 7, 2, 9, 1, 5, 3, 8, 6])
    scores = ELBOW[perm]  # unsorted input
    layers, rule = select_vital_layers(scores)
    assert rule == "elbow"
    assert layers == sorted(int(torch.nonzero(perm == i)) for i in (0, 1, 2))


def test_select_vital_layers_top20_fallback_without_elbow():
    scores = torch.tensor([5.0, 10.0, 1.0, 8.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0])  # linear after sorting
    assert select_elbow(scores) is None
    layers, rule = select_vital_layers(scores)
    assert rule == "top_0.2" and layers == [1, 5]  # round(0.2 * 10) = 2 layers
    assert select_top_fraction(torch.tensor([3.0, 1.0, 2.0]), 0.2) == [0]  # at least one layer


def test_select_vital_layers_elbow_needs_ci_separation():
    wide_lo, wide_hi = ELBOW - 0.5, ELBOW + 0.5
    layers, rule = select_vital_layers(ELBOW, lo=wide_lo, hi=wide_hi)
    assert rule == "top_0.2" and layers == [0, 1]
    layers, rule = select_vital_layers(ELBOW, lo=ELBOW - 0.01, hi=ELBOW + 0.01)
    assert rule == "elbow" and layers == [0, 1, 2]


def test_split_half_spearman():
    per_prompt = torch.arange(6, dtype=torch.float64)[:, None] + 0.01 * torch.arange(16, dtype=torch.float64)[None]
    rho = split_half_spearman(per_prompt, halves=(range(8), range(8, 16)))
    assert rho.shape == (1,) and rho.item() == pytest.approx(1.0)
    rhos = split_half_spearman(per_prompt, n_splits=3, generator=torch.Generator().manual_seed(0))
    assert rhos.shape == (3,) and torch.allclose(rhos, torch.ones(3, dtype=rhos.dtype))
    with pytest.raises(ValueError, match="disjoint"):
        split_half_spearman(per_prompt, halves=([0, 1, 2], [2, 3, 4]))
    assert spearman_rho(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_paired_bootstrap_ci_is_paired():
    x = torch.randn(16, generator=torch.Generator().manual_seed(1)).double()
    per_prompt = torch.stack([x, x + 1.0, torch.full((16,), 0.5, dtype=torch.float64)])
    mean, lo, hi = paired_bootstrap_ci(per_prompt, n_boot=500, generator=torch.Generator().manual_seed(2))
    assert mean.shape == lo.shape == hi.shape == (3,)
    assert maxdiff(mean, per_prompt.mean(-1)) < 1e-12
    assert bool((lo <= mean).all() and (mean <= hi).all())
    # same resampled prompt indices for every layer: a constant offset survives exactly
    assert lo[1].item() - lo[0].item() == pytest.approx(1.0, abs=1e-9)
    assert hi[1].item() - hi[0].item() == pytest.approx(1.0, abs=1e-9)
    assert lo[2].item() == pytest.approx(0.5) and hi[2].item() == pytest.approx(0.5)


# =========================================================================== E2-2 Cont^(l) via wo(x)


class QuantLikeLinear(torch.nn.Module):
    """int8 weight + per-row scale, dequantised inside ``forward`` (no ``.weight`` attribute).

    Any code that tried ``wo.weight @ x`` instead of calling ``wo(x)`` raises ``AttributeError``.
    """

    def __init__(self, din: int, dout: int, seed: int = 0) -> None:
        super().__init__()
        w = torch.randn(dout, din, generator=torch.Generator().manual_seed(seed))
        scale = w.abs().amax(1, keepdim=True) / 127
        self.register_buffer("qweight", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale)
        self.calls = 0

    def dequant(self):
        return self.qweight.double() * self.scale.double()

    def forward(self, x):
        self.calls += 1
        return x @ (self.qweight.to(x.dtype) * self.scale.to(x.dtype)).t()


def test_quant_like_wrapper_has_no_weight():
    with pytest.raises(AttributeError):
        _ = QuantLikeLinear(8, 4).weight


@pytest.mark.parametrize("query_chunk,key_chunk", [(1024, None), (5, 3)])
def test_contribution_tracker_calls_wo_module(capabilities, layouts, make_qkv, make_ctx, dense_attention, native_sdpa,
                                              query_chunk, key_chunk):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout, batch=4)
    ctx = make_ctx((UNCOND, COND), chunk=2)  # cond rows 2, 3
    span = TokenSpan(2, 5)
    Hq, d, D = 8, 16, 16
    wo = QuantLikeLinear(Hq * d, D, seed=3)
    g = torch.Generator().manual_seed(4)
    gate_logits = torch.randn(4, layout.n_query, Hq * d, generator=g)
    g1 = torch.randn(4, 1, D, generator=g)

    tracker = ContributionTracker(span, lambda block: wo)
    sess = XAISession(capabilities["moss"], query_chunk=query_chunk, key_chunk=key_chunk)
    sess.add_observer(tracker.factory)
    sess.attention(q, k, v, block=1, ctx=ctx, layout=layout, native=native_sdpa(),
                   extras={"gate_logits": gate_logits, "g1": g1})
    assert wo.calls == 2  # raw + gated, both through the module call

    rows = [2, 3]
    p, _ = dense_attention(q, k)
    ks = layout.txt_keys(span)
    A = p[rows][:, :, layout.q_img, ks]                                   # [2, Hq, N, |I|] of the full softmax
    vv = v.double()[rows][:, :, ks].repeat_interleave(4, dim=1)            # h_kv = h_q // 4
    o = (A @ vv).transpose(1, 2).flatten(2)                               # concat_h -> [2, N, Hq*d]
    W = wo.dequant()
    raw_ref = (o @ W.t()).norm(dim=-1).mean(-1)
    gate = torch.sigmoid(gate_logits.double()[rows][:, layout.q_img])
    gated_ref = (((o * gate) @ W.t()) * g1.double()[rows]).norm(dim=-1).mean(-1)

    raw, gated = tracker.scores("raw_mvo"), tracker.scores("gated_mvo")
    assert set(raw) == {1} and raw[1].shape == (2,)
    torch.testing.assert_close(raw[1].double(), raw_ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(gated[1].double(), gated_ref, rtol=1e-4, atol=1e-6)
    assert maxdiff(raw[1], gated[1]) > 1e-4


def test_contribution_from_context_raw_vs_gated():
    g = torch.Generator().manual_seed(5)
    o = torch.randn(2, 16, 128, generator=g)
    wo = QuantLikeLinear(128, 16, seed=6)
    open_gate = contribution_from_context(o, wo, gate_logits=torch.full((2, 16, 128), 50.0))
    assert maxdiff(open_gate["raw_mvo"], open_gate["gated_mvo"]) < 1e-5  # sigmoid(50) == 1, g1 absent
    half = contribution_from_context(o, wo, gate_logits=torch.zeros(2, 16, 128))
    assert maxdiff(half["gated_mvo"], 0.5 * half["raw_mvo"]) < 1e-5  # wo is linear
    assert "gated_mvo" not in contribution_from_context(o, wo)


def test_attention_contribution_matches_manual(dense_attention):
    g = torch.Generator().manual_seed(7)
    a = torch.rand(2, 8, 16, 3, generator=g)
    v = torch.randn(2, 2, 3, 16, generator=g)
    wo = QuantLikeLinear(128, 16, seed=8)
    got = attention_contribution(a, v, wo)["raw_mvo"]
    o = (a.double() @ v.double().repeat_interleave(4, dim=1)).transpose(1, 2).flatten(2)
    torch.testing.assert_close(got.double(), (o @ wo.dequant().t()).norm(dim=-1).mean(-1), rtol=1e-4, atol=1e-6)


def test_top_bottom_random_sets():
    scores = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0])
    sets = top_bottom_random(scores, generator=torch.Generator().manual_seed(0))
    assert sets["top"] == [1, 6] and sets["bottom"] == [0, 9]
    assert len(sets["random"]) == 2 and set(sets["random"]) <= set(range(10))
    again = top_bottom_random(scores, generator=torch.Generator().manual_seed(0))
    assert again["random"] == sets["random"]


def test_relative_depth_blocks():
    assert relative_depth_blocks(28) == [7, 14, 21]
    assert relative_depth_blocks(42) == [10, 21, 31]
