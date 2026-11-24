"""Direction 01 heatmap invariants (``plan/m1.md`` §3 E1; ``docs/specs/01_heatmap.md``).

E1-1 cond-only streaming ``img_to_txt``, average-rank Top-K, 2D Gaussian smoothing;
E1-2 scale-free concept scores, concept softmax order, Krea 2 ``pre_gate`` vs ``post_gate``;
E1-3 pre-GQA-repeat V-norm ratio and span-mass-preserving 1D text-axis blur.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed; run on a torch host")

from xai_dit.adapters.base import COND, UNCOND, InterventionSpec, TokenSpan  # noqa: E402
from xai_dit.core.heatmap import (  # noqa: E402
    ConceptAccumulator,
    ImgToTxtHeatmap,
    attention_readout,
    average_rank_top_k,
    concept_scores,
    concept_softmax,
    concept_stream_attention,
    fallback_middle_blocks,
    gaussian_kernel_2d,
    gaussian_smooth_2d,
    img_to_txt,
    prompt_output_space_scores,
    relative_threshold,
    span_mass,
    symmetric_kl,
    text_axis_blur_factory,
    v_norm_ratio,
    word_map,
)
from xai_dit.core.layer_selection import rankdata  # noqa: E402
from xai_dit.core.session import (  # noqa: E402
    TextAxisBlur,
    XAISession,
    chunked_attention,
    expand_kv,
    gaussian_kernel_1d,
)


def maxdiff(a, b) -> float:
    return (a.double() - b.double()).abs().max().item()


# =========================================================================== E1-1 ExploringMMDiT


def test_img_to_txt_heatmap_is_cond_only(capabilities, layouts, make_qkv, make_ctx, dense_attention, native_sdpa):
    """Cond rows are sliced *before* the head mean (no CFG mixing); streamed over q/k chunks."""
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout, batch=4)
    ctx = make_ctx((UNCOND, COND), chunk=2)  # cond rows are 2, 3
    hm = ImgToTxtHeatmap()
    sess = XAISession(capabilities["moss"], query_chunk=5, key_chunk=3)
    sess.add_observer(hm.factory)
    sess.attention(q, k, v, block=0, ctx=ctx, layout=layout, native=native_sdpa())
    p, _ = dense_attention(q, k)
    i2t = p[:, :, layout.q_img, layout.k_txt]
    got = hm.maps[0].value()
    assert got.shape == (2, 1, layout.n_img, layout.n_txt)
    assert maxdiff(got, i2t[[2, 3]].mean(1, keepdim=True)) < 1e-5
    assert maxdiff(got, i2t[[0, 1]].mean(1, keepdim=True)) > 1e-3  # would differ if uncond leaked in


def test_img_to_txt_heatmap_mean_over_steps_and_per_head(capabilities, layouts, make_qkv, make_ctx, dense_attention,
                                                          native_sdpa):
    layout = layouts["txt_img"]
    hm = ImgToTxtHeatmap(head_mean=False)
    sess = XAISession(capabilities["krea2"], query_chunk=7)
    sess.add_observer(hm.factory, blocks=[1])
    refs = []
    for step, sigma in enumerate((1.0, 0.875)):
        q, k, v = make_qkv(layout, seed=step)
        ctx = make_ctx((COND, UNCOND), chunk=1, sigma=sigma)
        sess.attention(q, k, v, block=1, ctx=ctx, layout=layout, native=native_sdpa())
        sess.attention(q, k, v, block=0, ctx=ctx, layout=layout, native=native_sdpa())  # not observed
        p, _ = dense_attention(q, k)
        refs.append(p[[0], :, layout.q_img, layout.k_txt])
    assert set(hm.maps) == {1}
    got = hm.maps[1].value()
    assert got.shape == (1, 8, layout.n_img, layout.n_txt)
    assert maxdiff(got, (refs[0] + refs[1]) / 2) < 1e-5
    assert maxdiff(hm.block_mean(), got) == 0


def test_img_to_txt_slice_is_order_invariant(layouts, prob_collector):
    g = torch.Generator().manual_seed(2)
    B, H, N, L, d = 1, 8, 16, 8, 16
    qi, qt, ki, kt = (torch.randn(B, H, n, d, generator=g) for n in (N, L, N, L))
    it, ti = layouts["img_txt"], layouts["txt_img"]
    a, b = prob_collector((B, H, N + L, N + L)), prob_collector((B, H, N + L, N + L))
    chunked_attention(torch.cat([qi, qt], 2), torch.cat([ki, kt], 2), None, observers=[a])
    chunked_attention(torch.cat([qt, qi], 2), torch.cat([kt, ki], 2), None, observers=[b])
    assert maxdiff(img_to_txt(a.probs, it), img_to_txt(b.probs, ti)) < 1e-6
    assert img_to_txt(a.probs, it).sum(-1).max() < 1.0  # joint softmax: text share < 1


def test_rankdata_ties_averaged():
    assert rankdata(torch.tensor([1.0, 1.0, 2.0])).tolist() == [1.5, 1.5, 3.0]
    assert rankdata(torch.tensor([3.0, 1.0, 2.0])).tolist() == [3.0, 1.0, 2.0]


def test_average_rank_top_k():
    metrics = {"bce": [0.5, 0.1, 0.3, 0.2], "soft_miou": [0.1, 0.9, 0.5, 0.6], "mse": [0.4, 0.1, 0.3, 0.2]}
    higher = {"soft_miou": True}
    assert average_rank_top_k(metrics, higher, k=2) == [1, 3]
    assert average_rank_top_k(metrics, higher, k=1) == [1]
    # flipping the direction of soft_miou changes the consensus
    assert average_rank_top_k({"soft_miou": metrics["soft_miou"]}, {}, k=1) == [0]


def test_fallback_middle_blocks():
    assert fallback_middle_blocks(10) == list(range(2, 8))
    assert fallback_middle_blocks(28) == list(range(6, 22))


def test_gaussian_kernels():
    k1 = gaussian_kernel_1d(7, 1.0)
    assert abs(k1.sum().item() - 1) < 1e-6 and torch.equal(k1, k1.flip(0)) and k1.argmax() == 3
    assert torch.equal(gaussian_kernel_1d(5, 0.0), torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        gaussian_kernel_1d(4, 1.0)
    k2 = gaussian_kernel_2d(7, 1.0)
    assert k2.shape == (7, 7) and abs(k2.sum().item() - 1) < 1e-6
    assert maxdiff(k2, k2.T) == 0 and k2.argmax() == 24


def test_gaussian_smooth_2d_delta_response_and_mass():
    h = w = 9
    maps = torch.zeros(2, 3, h * w)
    maps[..., 4 * w + 4] = 1.0
    out = gaussian_smooth_2d(maps, h, w)  # k=7, sigma=1, zero padding
    assert out.shape == maps.shape
    assert maxdiff(out.sum(-1), torch.ones(2, 3)) < 1e-6  # interior delta keeps its mass
    grid = out.reshape(2, 3, h, w)
    assert maxdiff(grid[1, 2, 1:8, 1:8], gaussian_kernel_2d(7, 1.0)) < 1e-7
    assert int(out[0, 0].argmax()) == 4 * w + 4
    # zero padding: a corner delta loses mass at the border
    corner = torch.zeros(1, h * w)
    corner[0, 0] = 1.0
    assert gaussian_smooth_2d(corner, h, w).sum() < 1.0


def test_relative_threshold_and_word_map():
    maps = torch.tensor([[0.1, 0.5, 1.0], [2.0, 0.2, 0.9]])
    assert relative_threshold(maps, 0.4).tolist() == [[False, True, True], [True, False, True]]
    m = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
    assert maxdiff(word_map(m, TokenSpan(1, 3)), m[..., 1:3].mean(-1)) == 0
    assert maxdiff(word_map(m, [0, 4]), m[..., [0, 4]].mean(-1)) == 0


# =========================================================================== E1-2 ConceptAttention


def test_concept_scores_are_scale_free_dot_products():
    g = torch.Generator().manual_seed(4)
    o_c, o_x = torch.randn(2, 3, 32, generator=g), torch.randn(2, 16, 32, generator=g)
    s = concept_scores(o_c, o_x)
    assert s.shape == (2, 3, 16)
    assert maxdiff(s, o_c @ o_x.transpose(1, 2)) < 1e-4  # no 1/sqrt(D)
    assert maxdiff(concept_scores(2 * o_c, o_x), 2 * s) < 1e-4


@pytest.mark.parametrize("order", ["per_layer", "after_mean"])
def test_concept_softmax_and_streaming_accumulator(order):
    stack = torch.randn(4, 2, 3, 16, generator=torch.Generator().manual_seed(5))
    ref = concept_softmax(stack, order, temperature=0.5)
    assert ref.shape == (2, 3, 16)
    assert maxdiff(ref.sum(-2), torch.ones(2, 16)) < 1e-6  # softmax over the concept axis
    acc = ConceptAccumulator(order, temperature=0.5)
    for layer in stack:
        acc.add(layer)
    assert maxdiff(acc.value(), ref) < 1e-5
    with pytest.raises(ValueError):
        concept_softmax(stack, "before_mean")


def test_concept_softmax_orders_differ():
    stack = torch.randn(4, 1, 3, 8, generator=torch.Generator().manual_seed(6)) * 3
    assert maxdiff(concept_softmax(stack, "per_layer"), concept_softmax(stack, "after_mean")) > 1e-3


def test_krea2_pre_gate_vs_post_gate_readout():
    """``attn_pre_proj`` (ctx) and ``post_gate`` (``ctx * gate_value``) are distinct readouts."""
    g = torch.Generator().manual_seed(7)
    ctx, gl = torch.randn(2, 24, 128, generator=g), torch.randn(2, 24, 128, generator=g)
    pre = attention_readout(ctx, gl, "pre_gate")
    post = attention_readout(ctx, gl, "post_gate")
    assert torch.equal(pre, ctx)
    assert maxdiff(post, ctx * torch.sigmoid(gl)) < 1e-6
    assert maxdiff(post, ctx) > 1e-2
    assert maxdiff(post, ctx * gl) > 1e-2  # gate acts post-sigmoid, never on raw logits
    with pytest.raises(ValueError):
        attention_readout(ctx, None, "post_gate")
    # downstream concept maps differ between the two readouts
    o_c = torch.randn(2, 3, 128, generator=g)
    assert maxdiff(concept_scores(o_c, pre), concept_scores(o_c, post)) > 1e-2


def test_prompt_output_space_scores(layouts):
    o = torch.randn(2, 24, 32, generator=torch.Generator().manual_seed(8))
    span = TokenSpan(2, 5)
    for order in ("img_txt", "txt_img"):
        lay = layouts[order]
        s = prompt_output_space_scores(o, lay, span)
        assert s.shape == (2, 3, 16)
        assert maxdiff(s, concept_scores(o[:, lay.txt_queries(span)], o[:, lay.q_img])) == 0
    with pytest.raises(ValueError):
        prompt_output_space_scores(o[:, :16], layouts["cross"], span)


def test_concept_stream_attention_matches_dense(dense_attention):
    g = torch.Generator().manual_seed(9)
    q_c = torch.randn(2, 8, 3, 16, generator=g)
    k_x, v_x = torch.randn(2, 2, 24, 16, generator=g), torch.randn(2, 2, 24, 16, generator=g)
    k_c, v_c = torch.randn(2, 2, 3, 16, generator=g), torch.randn(2, 2, 3, 16, generator=g)
    out = concept_stream_attention(q_c, k_x, v_x, k_c, v_c)
    _, ref = dense_attention(q_c, torch.cat([k_x, k_c], 2), torch.cat([v_x, v_c], 2))
    assert out.shape == (2, 8, 3, 16) and maxdiff(out, ref) < 1e-5


# =========================================================================== E1-3 Seg4Diff


def test_v_norm_ratio_uses_pre_repeat_values():
    B, Hkv, T, N, d = 2, 2, 5, 16, 16
    v_txt = torch.full((B, Hkv, T, d), 2.0)
    v_img = torch.full((B, Hkv, N, d), 1.0)
    r = v_norm_ratio(v_txt, v_img, num_kv_heads=Hkv)
    assert r["txt_norm"].item() == pytest.approx(2 * math.sqrt(d))
    assert r["img_norm"].item() == pytest.approx(math.sqrt(d))
    assert r["ratio"].item() == pytest.approx(2.0)
    assert r["per_head_ratio"].shape == (Hkv,) and torch.allclose(r["per_head_ratio"], torch.full((Hkv,), 2.0))
    rc = v_norm_ratio(v_txt, v_img, num_kv_heads=Hkv, mode="concat")
    assert rc["ratio"].item() == pytest.approx(2.0)
    # Passing GQA-repeated V is rejected; repeating would inflate the concatenated norm by sqrt(g).
    with pytest.raises(ValueError, match="before GQA repeat"):
        v_norm_ratio(expand_kv(v_txt, 4), expand_kv(v_img, 4), num_kv_heads=Hkv)
    inflated = v_norm_ratio(expand_kv(v_txt, 4), expand_kv(v_img, 4), mode="concat")
    assert inflated["txt_norm"].item() == pytest.approx(2 * rc["txt_norm"].item())
    with pytest.raises(ValueError):
        v_norm_ratio(v_txt, v_img, mode="max")


def test_symmetric_kl():
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.4, 0.4, 0.2])
    assert symmetric_kl(p, p).item() == pytest.approx(0.0, abs=1e-9)
    assert symmetric_kl(p, q).item() > 0
    assert symmetric_kl(p, q).item() == pytest.approx(symmetric_kl(q, p).item())


def test_text_axis_blur_preserves_span_mass(capabilities, layouts, make_qkv, make_ctx, prob_collector, native_sdpa):
    """1D blur along text keys inside ``S`` only: row sums == 1, span mass preserved, everything else exact."""
    layout = layouts["img_txt"]
    span = TokenSpan(2, 6)
    q, k, v = make_qkv(layout)
    spec = InterventionSpec("attn_override", params={"override": text_axis_blur_factory(span)})
    sess = XAISession(capabilities["moss"], interventions=[spec])
    store: list = []
    sess.add_observer(prob_collector.factory(store))
    sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native_sdpa())
    blurred = store[0].probs
    base = prob_collector(blurred.shape)
    chunked_attention(q, k, None, observers=[base])
    base = base.probs

    ks = layout.txt_keys(span)
    outside = torch.ones(layout.n_key, dtype=torch.bool)
    outside[ks] = False
    assert maxdiff(blurred.sum(-1), torch.ones(blurred.shape[:3])) < 1e-6
    assert maxdiff(span_mass(blurred, ks), span_mass(base, ks)) < 1e-6
    assert maxdiff(blurred[..., outside], base[..., outside]) == 0
    assert maxdiff(blurred[:, :, layout.q_txt], base[:, :, layout.q_txt]) == 0  # text queries untouched
    assert maxdiff(blurred[:, :, layout.q_img, ks], base[:, :, layout.q_img, ks]) > 1e-4


def test_text_axis_blur_strength_zero_and_head_selection(layouts):
    layout = layouts["cross"]
    p = torch.softmax(torch.randn(2, 8, layout.n_query, layout.n_key, generator=torch.Generator().manual_seed(10)), -1)
    span = TokenSpan(0, 5)
    q_sl = slice(0, layout.n_query)
    assert TextAxisBlur.for_layout(layout, span, strength=0.0)(p, q_sl) is p
    blur = text_axis_blur_factory(span, heads=[3], num_heads=8)(layout, None)
    out = blur(p, q_sl)
    others = [h for h in range(8) if h != 3]
    assert maxdiff(out[:, others], p[:, others]) == 0
    assert maxdiff(out[:, 3, :, 0:5].sum(-1), p[:, 3, :, 0:5].sum(-1)) < 1e-6
    assert maxdiff(out[:, 3], p[:, 3]) > 1e-4
    with pytest.raises(ValueError):
        text_axis_blur_factory(span, heads=[3])(layout, None)
