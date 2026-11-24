"""Direction 04 MI invariants (``plan/m1.md`` §3 E4; ``docs/specs/04_verification_mi.md``).

E4-1 two-pass dynamic sink masking (``residual_ratio < 1e-6`` at the *pre-selected* indices,
Anima grouped pad span); E4-2 ``R -> I`` pre-softmax masking isolation + Anima N/A; E4-3
``gate_logits`` vs ``gate_value`` and post-sigmoid ``gate_override``; E4-4 post-softmax per-head
``A_j V_j`` ablation without GQA sibling bleed; E4-5 ``residual_out`` tap.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed; run on a torch host")

from xai_dit.adapters.base import COND, UNCOND, CaptureSpec, TokenMap, TokenSpan  # noqa: E402
from xai_dit.core.heatmap import attention_readout  # noqa: E402
from xai_dit.core.session import (  # noqa: E402
    XAISession,
    chunked_attention,
    dynamic_sink_edit,
    index_mask,
    select_dynamic_sinks,
    union_mask,
)
from xai_dit.core.verification import (  # noqa: E402
    MeanGate,
    SpanMassTracker,
    StructurallyNotApplicable,
    apply_gate_override,
    dose_response_fit,
    early_step_index,
    entropy,
    expected_logit_bias_mass,
    gate_stats,
    gate_value,
    img_to_txt_text_softmax,
    masking_isolation_report,
    materialize_probs,
    normalized_register_share,
    pre_dit_span_swap,
    prob_scale,
    register_mask_spec,
    replace_sink_values,
    residual_tap,
    sink_mask_spec,
    sink_masking_report,
    span_masses,
    static_sink_logit_bias,
    token_head_ablation_spec,
)


def maxdiff(a, b) -> float:
    return (a.double() - b.double()).abs().max().item()


def _anima_map(L: int = 8, Lt: int = 5) -> TokenMap:
    ids = list(range(100, 100 + Lt - 1)) + [1] + [0] * (L - Lt)
    return TokenMap.from_lengths(ids, suffix=1, pad=L - Lt)  # S=[0,4) R=[4,5) P=[5,8)


def _moss_map(L: int = 8) -> TokenMap:
    return TokenMap.from_lengths(list(range(200, 200 + L)), prefix=2, suffix=3)  # R_prefix=[0,2) S=[2,5) R=[5,8)


# =========================================================================== E4-1 dynamic sinks


@pytest.mark.parametrize("query_chunk,key_chunk", [(1024, None), (5, 3)])
def test_two_pass_sink_masking_residual_ratio(layouts, make_qkv, dense_attention, query_chunk, key_chunk):
    layout = layouts["img_txt"]
    q, k, _ = make_qkv(layout)
    rep = sink_masking_report(q, k, layout, top_k=2, query_chunk=query_chunk, key_chunk=key_chunk)
    sel = rep["selection"]
    assert sel.unit_ids.shape == (2, 8, 2) and sel.mask.shape == (2, 8, layout.n_key)
    assert bool((sel.mask.sum(-1) == 2).all())
    assert bool((rep["mass_before"] > 0).all())
    assert bool((rep["residual_ratio"] < 1e-6).all())

    # pass 1 picks the top-k incoming-mass keys over image queries (independent dense reference)
    p, _ = dense_attention(q, k)
    mass = p[:, :, layout.q_img].mean(2)
    top3 = mass.topk(3, dim=-1).values
    if bool(((top3[..., 1] - top3[..., 2]) > 1e-5).all()):  # skip comparison only on near-ties
        assert torch.equal(sel.unit_ids.sort(-1).values, mass.topk(2, dim=-1).indices.sort(-1).values)

    # mass_after is re-read at the pre-selected indices (not re-selected after masking)
    q_img = index_mask(layout.n_query, layout.q_img)
    after = materialize_probs(q, k, logit_edits=[dynamic_sink_edit(sel, q_img)])
    m_after = after[:, :, layout.q_img].mean(2)
    assert maxdiff(rep["mass_after"], (m_after * sel.mask).sum(-1)) < 1e-7
    # text query rows are not edited by the pass-2 mask
    base = materialize_probs(q, k)
    assert maxdiff(after[:, :, layout.q_txt], base[:, :, layout.q_txt]) == 0


def test_union_selection_shares_indices_across_heads(layouts, make_qkv):
    q, k, _ = make_qkv(layouts["img_txt"])
    sel = sink_masking_report(q, k, layouts["img_txt"], top_k=3, selection="union")["selection"]
    assert torch.equal(sel.mask, sel.mask[:, :1].expand_as(sel.mask))


def test_select_dynamic_sinks_groups_tied_pad_span():
    mass = torch.full((1, 2, 8), 0.01)
    mass[..., 0] = 0.2
    mass[..., 5:8] = 0.1  # Anima P: tied zero-pad logits
    sel = select_dynamic_sinks(mass, 1, groups=[slice(5, 8)])
    assert bool(sel.mask[..., 5:8].all()) and not bool(sel.mask[..., :5].any())
    assert bool((sel.unit_ids == 5).all())
    single = select_dynamic_sinks(mass, 1)
    assert single.mask[..., 0].all() and int(single.mask.sum()) == 2
    rnd = select_dynamic_sinks(mass, 2, groups=[slice(5, 8)], random=True, generator=torch.Generator().manual_seed(0))
    assert not bool(rnd.mask[..., 6:8].any() & ~rnd.mask[..., 5:6].any())  # pad group still all-or-none


def test_anima_two_pass_sink_mask_session_grouped_pad(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["cross"]
    tm = _anima_map()
    P = tm.spans["P"]
    q, k, v = make_qkv(layout)
    k[:, :, P.start:P.end] = k[:, :, P.start:P.start + 1].clone()  # zero ctx -> identical pad keys
    v[:, :, P.start:P.end] = v[:, :, P.start:P.start + 1].clone()
    spec = sink_mask_spec(1, blocks=[0], groups=[P])
    assert spec.hook == "two_pass_sink_mask" and spec.params["groups"] == (P,)
    sess = XAISession(capabilities["anima"], interventions=[spec])
    out = sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native_sdpa())
    (rec,) = sess.sink_log
    m = rec.selection.mask
    pad = m[..., P.start:P.end]
    assert bool((pad.all(-1) | ~pad.any(-1)).all())  # pad slots masked together or not at all
    ref = chunked_attention(q, k, v, logit_edits=[dynamic_sink_edit(rec.selection, index_mask(layout.n_query, layout.q_img))])
    assert maxdiff(out, ref) < 1e-6
    rep = sink_masking_report(q, k, layout, 1, groups=[P])
    assert bool((rep["residual_ratio"] < 1e-6).all())


def test_static_sink_edits():
    p = torch.softmax(torch.randn(2, 4, 6, 10, generator=torch.Generator().manual_seed(1)), -1)
    sink = torch.zeros(10, dtype=torch.bool)
    sink[[0, 3]] = True
    for eta in (0.25, 2.0):
        scaled = prob_scale(p, sink, eta)
        assert maxdiff(scaled.sum(-1), torch.ones(2, 4, 6)) < 1e-6
        assert maxdiff(scaled[..., sink].sum(-1), eta * p[..., sink].sum(-1)) < 1e-6
        logits = torch.log(p)
        biased = torch.softmax(static_sink_logit_bias(sink, eta)(logits, slice(0, 6), slice(0, 10)), -1)
        assert maxdiff(biased[..., sink].sum(-1), expected_logit_bias_mass(p[..., sink].sum(-1), eta)) < 1e-6
    with pytest.raises(ValueError):
        static_sink_logit_bias(sink, 0.0)
    assert entropy(torch.tensor([0.0, 1.0, 0.0])).item() == 0
    assert entropy(torch.full((4,), 0.25)).item() == pytest.approx(math.log(4))


# =========================================================================== E4-2 template registers


@pytest.mark.parametrize("order", ["img_txt", "txt_img"])
@pytest.mark.parametrize("target", ["img", "S"])
def test_register_masking_isolation_report(layouts, make_qkv, order, target):
    layout = layouts[order]
    tm = _moss_map()
    q, k, _ = make_qkv(layout)
    key_target = "img" if target == "img" else [tm.spans["S"]]
    rep = masking_isolation_report(q, k, layout, [tm.spans["R_prefix"], tm.spans["R"]], key_target)
    assert rep["masked_prob_max"] < 1e-6
    assert rep["logit_outside_max_abs_diff"] == 0
    assert rep["register_row_sum_err"] < 1e-6
    assert rep["other_rows_max_abs_diff"] < 1e-6


def test_register_to_image_session_masking(capabilities, layouts, make_qkv, make_ctx, prob_collector, native_sdpa):
    layout = layouts["img_txt"]
    tm = _moss_map()
    q, k, v = make_qkv(layout)
    spec = register_mask_spec(capabilities["moss"], tm, "img", blocks=[0])
    assert spec.hook == "span_query_mask" and spec.params["key_target"] == "img"
    sess = XAISession(capabilities["moss"], interventions=[spec])
    store: list = []
    sess.add_observer(prob_collector.factory(store))
    sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native_sdpa())
    P = store[0].probs
    base = prob_collector(P.shape)
    chunked_attention(q, k, None, observers=[base])
    base = base.probs
    r_rows = union_mask(layout.n_query, [layout.txt_queries(tm.spans[n]) for n in ("R_prefix", "R")])
    assert int(r_rows.sum()) == 5
    assert P[:, :, r_rows][..., layout.k_img].max().item() < 1e-6        # R -> I < 1e-6
    assert maxdiff(P[:, :, r_rows].sum(-1), torch.ones(2, 8, 5)) < 1e-6  # R row sum == 1
    assert maxdiff(P[:, :, ~r_rows], base[:, :, ~r_rows]) < 1e-6         # other query rows unchanged


def test_register_masking_not_applicable_on_anima(capabilities):
    with pytest.raises(StructurallyNotApplicable):
        register_mask_spec(capabilities["anima"], _anima_map(), "img", blocks=[0])
    assert issubclass(StructurallyNotApplicable, ValueError)
    spec = register_mask_spec(capabilities["krea2"], TokenMap.from_lengths(list(range(9)), suffix=5), "S", blocks=[1])
    assert spec.params["query_span"] == (TokenSpan(4, 9),) and spec.params["key_target"] == TokenSpan(0, 4)


def test_span_mass_decomposition(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    # Anima (cross-attn): S + R + P == 1
    lay = layouts["cross"]
    tm = _anima_map()
    q, k, v = make_qkv(lay)
    p = materialize_probs(q, k)
    masses = span_masses(p[:, :, lay.q_img].mean(2), lay, tm)
    assert maxdiff(masses["S"] + masses["R"] + masses["P"], torch.ones(2, 8)) < 1e-6
    share = normalized_register_share(masses)
    assert bool(((share >= 0) & (share <= 1)).all())

    tracker = SpanMassTracker(tm)
    sess = XAISession(capabilities["anima"])
    sess.add_observer(tracker.factory)
    sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=lay, native=native_sdpa())
    total = sum(tracker.maps[(n, 0)].value() for n in ("S", "R", "P"))
    assert total.shape == (1, 8) and maxdiff(total, torch.ones(1, 8)) < 1e-6  # cond row only

    # joint (MOSS): R_prefix + S + R == txt_total < 1
    lay = layouts["img_txt"]
    tm = _moss_map()
    q, k, _ = make_qkv(lay)
    p = materialize_probs(q, k)
    masses = span_masses(p[:, :, lay.q_img].mean(2), lay, tm)
    assert maxdiff(masses["R_prefix"] + masses["S"] + masses["R"], masses["txt_total"]) < 1e-6
    assert bool((masses["txt_total"] < 1).all())


def test_pre_dit_span_swap():
    states = torch.zeros(2, 8, 4)
    donors = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4)
    span = TokenSpan(2, 5)
    swapped = pre_dit_span_swap(states, span, donors[0])
    assert torch.equal(swapped[:, 2:5], donors[0, 2:5].expand(2, -1, -1)) and swapped[:, :2].abs().max() == 0
    mean = pre_dit_span_swap(states, span, donors, mode="mean")
    assert torch.equal(mean[0, 2:5], donors[:, 2:5].mean(0)) and states.abs().max() == 0


# =========================================================================== E4-3 output gate


def test_gate_logits_vs_gate_value_capture(capabilities, layouts, make_ctx):
    """``gate_logits`` (pre-sigmoid) and ``gate_value`` are separate capture fields; post_gate = ctx * gate_value."""
    g = torch.Generator().manual_seed(2)
    ctx_t, gl = torch.randn(2, 24, 128, generator=g), torch.randn(2, 24, 128, generator=g)
    sess = XAISession(capabilities["krea2"], capture=CaptureSpec(
        fields={"attn_pre_proj", "gate_logits", "gate_value", "attn_out"}, branches={"cond", "uncond"}))
    ctx = make_ctx()
    sess.record("attn_pre_proj", ctx_t, block=0, ctx=ctx)
    sess.record("gate_logits", gl, block=0, ctx=ctx)
    sess.record("gate_value", gate_value(gl), block=0, ctx=ctx)
    sess.record("residual_out", ctx_t, block=0, ctx=ctx)  # not requested -> ignored
    (entry,) = sess.captures.values()
    assert entry.present() == ("attn_pre_proj", "gate_logits", "gate_value") and entry.rows == (0, 1)
    assert bool((entry.gate_logits < 0).any())
    assert bool(((entry.gate_value > 0) & (entry.gate_value < 1)).all())
    assert maxdiff(entry.gate_value, torch.sigmoid(entry.gate_logits)) == 0
    assert maxdiff(entry.gate_logits, entry.gate_value) > 0.1
    post = attention_readout(entry.attn_pre_proj, entry.gate_logits, "post_gate")
    assert maxdiff(post, entry.attn_pre_proj * entry.gate_value) < 1e-6


def test_gate_override_acts_post_sigmoid():
    g = torch.Generator().manual_seed(3)
    ctx, gl = torch.randn(2, 16, 32, generator=g), torch.randn(2, 16, 32, generator=g)
    gv = torch.sigmoid(gl)
    assert maxdiff(apply_gate_override(ctx, gl, "none"), ctx * gv) < 1e-6
    assert torch.equal(apply_gate_override(ctx, gl, "one"), ctx)
    closed = torch.full_like(gl, -20.0)
    assert torch.equal(apply_gate_override(ctx, closed, "one"), ctx)  # g' = 1, not sigmoid(1)
    assert maxdiff(apply_gate_override(ctx, closed, "one"), ctx * torch.sigmoid(torch.ones(1))) > 1e-2
    assert maxdiff(apply_gate_override(ctx, gl, "mean", 0.3), ctx * 0.3) < 1e-6
    assert maxdiff(apply_gate_override(ctx, gl, "const", torch.full((32,), 0.7)), ctx * 0.7) < 1e-6
    assert maxdiff(apply_gate_override(ctx, gl, "scale", 0.0), ctx * gv) < 1e-6
    assert maxdiff(apply_gate_override(ctx, gl, "scale", 1.0), ctx) < 1e-6
    assert maxdiff(apply_gate_override(ctx, gl, "scale", 0.5), ctx * (gv + 1) / 2) < 1e-6
    assert apply_gate_override(ctx.bfloat16(), gl, "one").dtype == torch.bfloat16
    with pytest.raises(ValueError):
        apply_gate_override(ctx, gl, "mean")
    with pytest.raises(ValueError):
        apply_gate_override(ctx, gl, "zero")


def test_gate_statistics_and_dose_response():
    gl = torch.randn(2, 24, 128, generator=torch.Generator().manual_seed(4))
    stats = gate_stats(gate_value(gl))
    assert 0 < stats["min"] <= stats["q10"] <= stats["q50"] <= stats["q90"] <= stats["max"] < 1
    mg = MeanGate()
    mg.add(0, gl)
    assert mg.value(0).shape == (128,)
    assert maxdiff(mg.value(0), torch.sigmoid(gl).flatten(0, 1).mean(0)) < 1e-6
    fit = dose_response_fit([0.0, 0.25, 0.5, 1.0], [1.0, 1.5, 2.0, 3.0])
    assert fit["slope"] == pytest.approx(2.0) and fit["intercept"] == pytest.approx(1.0)
    assert fit["spearman"] == pytest.approx(1.0)


# =========================================================================== E4-4 per-head token ablation


def test_per_head_token_ablation_no_gqa_sibling_bleed(capabilities, layouts, make_qkv, make_ctx, dense_attention,
                                                      native_sdpa):
    cap = capabilities["moss"]
    layout = layouts["img_txt"]
    span, head = TokenSpan(2, 4), 1
    q, k, v = make_qkv(layout)
    spec = token_head_ablation_spec([head], span, blocks=[0])
    assert spec.hook == "per_head_token_ablation" and spec.params["heads"] == (head,)
    sess = XAISession(cap, interventions=[spec])
    out = sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native_sdpa())
    base = chunked_attention(q, k, v)

    kv = cap.kv_head(head)
    siblings = [h for h in cap.q_heads_of(kv) if h != head]
    assert kv == 0 and siblings == [0, 2, 3]
    others = [h for h in range(8) if h != head]
    assert maxdiff(out[:, siblings], base[:, siblings]) == 0  # zero bleed into GQA siblings
    assert maxdiff(out[:, others], base[:, others]) == 0
    assert maxdiff(out[:, head, layout.q_txt], base[:, head, layout.q_txt]) == 0  # text queries untouched

    # image rows of the ablated head lose exactly sum_{j in span} A_hj V_{h_kv j} (no renormalisation)
    p, _ = dense_attention(q, k)
    ks = layout.txt_keys(span)
    term = p[:, head, layout.q_img, ks] @ v.double()[:, kv, ks]
    assert maxdiff(out[:, head, layout.q_img], base[:, head, layout.q_img].double() - term) < 1e-5
    assert term.abs().max() > 1e-4

    # contrast: a V edit on the shared KV head necessarily bleeds into every sibling
    key_mask = index_mask(layout.n_key, ks)
    v_edit = replace_sink_values(v, key_mask, "zero")
    bled = chunked_attention(q, k, v_edit)
    assert maxdiff(bled[:, siblings], base[:, siblings]) > 1e-4


def test_per_head_token_ablation_cond_only_scope(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["txt_img"]
    q, k, v = make_qkv(layout, batch=4)
    spec = token_head_ablation_spec([5], TokenSpan(0, 3), blocks=[0], cfg_scope="cond_only")
    sess = XAISession(capabilities["krea2"], interventions=[spec])
    out = sess.attention(q, k, v, block=0, ctx=make_ctx((UNCOND, COND), chunk=2), layout=layout,
                         native=native_sdpa())
    base = chunked_attention(q, k, v)
    assert maxdiff(out[:2], base[:2]) == 0       # uncond rows untouched
    assert maxdiff(out[2:, 5], base[2:, 5]) > 1e-4
    assert maxdiff(out[2:, [4, 6, 7]], base[2:, [4, 6, 7]]) == 0


# =========================================================================== E4-5 residual tap


@pytest.mark.parametrize("order,img_range", [("img_txt", (0, 16)), ("txt_img", (8, 24))])
def test_residual_tap_cond_rows_and_metadata(layouts, make_ctx, order, img_range):
    layout = layouts[order]
    hidden = torch.randn(4, layout.n_query, 32, generator=torch.Generator().manual_seed(5))
    keep = hidden.clone()
    ctx = make_ctx((UNCOND, COND), chunk=2, sigma=0.75)
    feats, metas = residual_tap(hidden, layout, ctx, block=3, grid=(4, 4))
    assert feats.shape == (2, 32, 4, 4)
    off = layout.q_img.start
    for b, row in enumerate((2, 3)):  # rows from cond_or_uncond metadata, not [0::2]
        for y, x in [(0, 0), (1, 2), (3, 3)]:
            assert torch.equal(feats[b, :, y, x], hidden[row, off + y * 4 + x])
    assert [(m.branch, m.sample_id, m.block, m.step_index) for m in metas] == [("cond", 0, 3, 2), ("cond", 1, 3, 2)]
    assert all(m.img_token_range == img_range and (m.grid_h, m.grid_w) == (4, 4) for m in metas)
    feats.zero_()
    assert torch.equal(hidden, keep)  # read-only tap
    unc, _ = residual_tap(hidden, layout, ctx, block=3, grid=(4, 4), branch="uncond")
    assert torch.equal(unc[0, :, 0, 0], hidden[0, off])
    with pytest.raises(ValueError):
        residual_tap(hidden, layout, ctx, block=3, grid=(2, 4))
    assert early_step_index(30) == 6 and early_step_index(8) == 2


def test_residual_out_capture_contract(capabilities, make_ctx):
    hidden = torch.randn(4, 24, 32, generator=torch.Generator().manual_seed(6))
    sess = XAISession(capabilities["moss"], capture=CaptureSpec(fields={"residual_out"}, blocks=[3]))
    ctx = make_ctx((UNCOND, COND), chunk=2)
    sess.record("residual_out", hidden, block=3, ctx=ctx)
    sess.record("residual_out", hidden, block=2, ctx=ctx)  # block not requested
    ((key, entry),) = sess.captures.items()
    assert key == (3, 0, (("cond", 0), ("cond", 1))) and entry.rows == (2, 3)
    assert torch.equal(entry.residual_out, hidden[2:4])


def test_img_to_txt_text_softmax_vs_joint_slice(layouts, make_qkv):
    layout = layouts["img_txt"]
    q, k, _ = make_qkv(layout)
    t = img_to_txt_text_softmax(q, k, layout)
    assert t.shape == (2, 8, 16, 8) and maxdiff(t.sum(-1), torch.ones(2, 8, 16)) < 1e-6
    joint = materialize_probs(q, k)[:, :, layout.q_img, layout.k_txt]
    assert bool((joint.sum(-1) < 1).all())
