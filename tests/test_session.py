"""Session engine invariants (``plan/m1.md`` §3: No-Op & Path, E1 chunking / order, E2 GQA, CFG policy)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed; run on a torch host")
import torch.nn.functional as F  # noqa: E402

from xai_dit.adapters.base import (  # noqa: E402
    COND,
    UNCOND,
    CaptureSpec,
    InterventionSpec,
    SeqLayout,
    StepWindow,
    TokenSpan,
)
from xai_dit.core.profiling import Profiler  # noqa: E402
from xai_dit.core.session import (  # noqa: E402
    _OOM,
    RunningMap,
    XAISession,
    chunked_attention,
    expand_kv,
    grouped_pv,
    grouped_scores,
    index_mask,
    slice_overlap,
    union_mask,
)


def maxdiff(a, b) -> float:
    return (a.double() - b.double()).abs().max().item()


# =========================================================================== chunked engine vs dense


@pytest.mark.parametrize("order", ["img_txt", "txt_img", "cross"])
@pytest.mark.parametrize("query_chunk,key_chunk", [(1024, None), (5, None), (1024, 3), (5, 3), (1, 1)])
def test_chunked_attention_matches_dense_reference(layouts, make_qkv, dense_attention, order, query_chunk, key_chunk):
    q, k, v = make_qkv(layouts[order])
    out = chunked_attention(q, k, v, query_chunk=query_chunk, key_chunk=key_chunk)
    _, ref = dense_attention(q, k, v)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert maxdiff(out, ref) < 1e-5


def test_query_chunked_vs_key_chunked_logsumexp_probs(layouts, make_qkv, prob_collector):
    layout = layouts["img_txt"]
    q, k, _ = make_qkv(layout)
    shape = (q.shape[0], q.shape[1], layout.n_query, layout.n_key)
    full, keyc = prob_collector(shape), prob_collector(shape)
    chunked_attention(q, k, None, observers=[full], query_chunk=7)
    chunked_attention(q, k, None, observers=[keyc], query_chunk=7, key_chunk=5)
    assert not torch.isnan(full.probs).any() and not torch.isnan(keyc.probs).any()  # every cell streamed once
    assert maxdiff(full.probs, keyc.probs) < 1e-6
    assert maxdiff(keyc.probs.sum(-1), torch.ones(shape[:3])) < 1e-6
    assert full.finished == keyc.finished == 1


def test_manual_path_matches_fused_sdpa_fp32(layouts, make_qkv):
    q, k, v = make_qkv(layouts["img_txt"])
    fused = F.scaled_dot_product_attention(q, expand_kv(k, 4), expand_kv(v, 4))
    for qc, kc in [(1024, None), (5, 3)]:
        assert maxdiff(chunked_attention(q, k, v, query_chunk=qc, key_chunk=kc), fused) < 1e-5


def test_manual_path_matches_fused_sdpa_bf16_within_1e3(layouts, make_qkv):
    """BF16 path identity ``<= 1e-3``. ``v`` is scaled so |out| <= max|v| keeps BF16 rounding below 1e-3."""
    q, k, v = make_qkv(layouts["img_txt"])
    qb, kb, vb = q.bfloat16(), k.bfloat16(), (v * 0.05).bfloat16()
    out = chunked_attention(qb, kb, vb, query_chunk=5)
    assert out.dtype == torch.bfloat16
    fused = F.scaled_dot_product_attention(qb.float(), expand_kv(kb.float(), 4), expand_kv(vb.float(), 4))
    assert maxdiff(out, fused) <= 1e-3


def test_attn_mask_bool_matches_sdpa(layouts, make_qkv):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    keep = torch.rand(layout.n_query, layout.n_key, generator=torch.Generator().manual_seed(3)) > 0.3
    keep[:, 0] = True
    ref = F.scaled_dot_product_attention(q, expand_kv(k, 4), expand_kv(v, 4), attn_mask=keep)
    for kc in (None, 5):
        assert maxdiff(chunked_attention(q, k, v, attn_mask=keep, key_chunk=kc, query_chunk=7), ref) < 1e-5


def test_oom_falls_back_to_key_chunked_logsumexp(layouts, make_qkv):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    full = slice(0, layout.n_key)
    seen = []

    def flaky(p, q_sl, k_sl):
        seen.append(k_sl)
        if k_sl == full:
            raise _OOM("simulated CUDA OOM")
        return p

    prof = Profiler()
    with prof.activate():
        out = chunked_attention(q, k, v, prob_edits=[flaky], oom_key_chunk=5)
    assert prof.counts.get("oom_fallback") == 1
    assert seen[0] == full and all(s.stop - s.start <= 5 for s in seen[1:])
    assert maxdiff(out, chunked_attention(q, k, v)) < 1e-5


def test_chunked_attention_argument_validation(layouts, make_qkv):
    q, k, v = make_qkv(layouts["img_txt"])
    with pytest.raises(ValueError):
        chunked_attention(q[0], k, v)
    with pytest.raises(ValueError):
        chunked_attention(q, k[:, :, :, :8], v)
    with pytest.raises(ValueError):
        chunked_attention(q, k, v, prob_override=lambda p, q_sl: p, key_chunk=4)
    with pytest.raises(ValueError):
        chunked_attention(q, k, v, query_chunk=0)


# =========================================================================== sequence order


def test_sequence_order_equivalence_img_txt_vs_txt_img(layouts, prob_collector):
    """``[img|txt]`` and ``[txt|img]`` give identical image/text outputs and ``img_to_txt`` maps."""
    g = torch.Generator().manual_seed(1)
    B, Hq, Hkv, N, L, d = 2, 8, 2, 16, 8, 16
    qi, qt = torch.randn(B, Hq, N, d, generator=g), torch.randn(B, Hq, L, d, generator=g)
    ki, kt = torch.randn(B, Hkv, N, d, generator=g), torch.randn(B, Hkv, L, d, generator=g)
    vi, vt = torch.randn(B, Hkv, N, d, generator=g), torch.randn(B, Hkv, L, d, generator=g)
    it, ti = layouts["img_txt"], layouts["txt_img"]
    shape = (B, Hq, N + L, N + L)
    c_it, c_ti = prob_collector(shape), prob_collector(shape)
    o_it = chunked_attention(torch.cat([qi, qt], 2), torch.cat([ki, kt], 2), torch.cat([vi, vt], 2),
                             observers=[c_it], query_chunk=5)
    o_ti = chunked_attention(torch.cat([qt, qi], 2), torch.cat([kt, ki], 2), torch.cat([vt, vi], 2),
                             observers=[c_ti], query_chunk=5, key_chunk=3)
    assert maxdiff(o_it[:, :, it.q_img], o_ti[:, :, ti.q_img]) < 1e-5
    assert maxdiff(o_it[:, :, it.q_txt], o_ti[:, :, ti.q_txt]) < 1e-5
    i2t_it = c_it.probs[:, :, it.q_img, it.k_txt]
    i2t_ti = c_ti.probs[:, :, ti.q_img, ti.k_txt]
    assert i2t_it.shape == (B, Hq, N, L)
    assert maxdiff(i2t_it, i2t_ti) < 1e-6


# =========================================================================== GQA


def test_gqa_h_kv_equals_h_q_div_4(capabilities, layouts, make_qkv):
    cap = capabilities["moss"]
    assert cap.gqa_group == 4 and [cap.kv_head(h) for h in range(8)] == [h // 4 for h in range(8)]
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    s = grouped_scores(q, k, 0.25)
    assert maxdiff(s, q @ expand_kv(k, 4).transpose(-1, -2) * 0.25) < 1e-5
    p = torch.softmax(s, -1)
    assert maxdiff(grouped_pv(p, v), p @ expand_kv(v, 4)) < 1e-5

    # Perturbing KV head 1 changes only Q heads 4..7 (never the siblings of KV head 0).
    k2, v2 = k.clone(), v.clone()
    k2[:, 1] += 1.0
    v2[:, 1] -= 2.0
    o1, o2 = chunked_attention(q, k, v), chunked_attention(q, k2, v2)
    assert maxdiff(o1[:, list(cap.q_heads_of(0))], o2[:, list(cap.q_heads_of(0))]) == 0
    assert maxdiff(o1[:, list(cap.q_heads_of(1))], o2[:, list(cap.q_heads_of(1))]) > 1e-3


# =========================================================================== exact no-op


_KO = {"keys": TokenSpan(0, 2)}
NOOP_CASES = {
    "no_hooks": lambda: {},
    "other_block": lambda: {"interventions": [InterventionSpec("logit_knockout", blocks=[3], params=_KO)]},
    "outside_window": lambda: {"interventions": [InterventionSpec("logit_knockout", window=StepWindow(start=5),
                                                                  params=_KO)]},
    "outside_sigma_window": lambda: {"interventions": [InterventionSpec("logit_knockout",
                                                                        window=StepWindow(sigma_max=0.5),
                                                                        params=_KO)]},
    "identity_logit_scale": lambda: {"interventions": [InterventionSpec("logit_scale",
                                                                        params={"gamma": 1.0, "bias": 0.0})]},
    "model_side_hook_only": lambda: {"interventions": [InterventionSpec("gate_override", params={"mode": "one"})]},
    "capture_other_block": lambda: {"capture": CaptureSpec(fields={"probs", "q"}, blocks=[2])},
}


@pytest.mark.parametrize("case", sorted(NOOP_CASES))
def test_inactive_hooks_are_exact_noop(case, capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    sess = XAISession(capabilities["moss"], **NOOP_CASES[case]())
    native = native_sdpa()
    out = sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native)
    assert native.calls == 1  # handed off to the unmodified kernel
    assert maxdiff(out, native(q, k, v)) == 0
    assert sess.captures == {} and sess.sink_log == []


def test_cond_only_hook_on_uncond_only_call_is_noop(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    spec = InterventionSpec("logit_knockout", cfg_scope="cond_only", params=_KO)
    sess = XAISession(capabilities["moss"], interventions=[spec])
    native = native_sdpa()
    out = sess.attention(q, k, v, block=0, ctx=make_ctx(order=(UNCOND,), chunk=2), layout=layout, native=native)
    assert native.calls == 1 and maxdiff(out, native(q, k, v)) == 0


def test_disabled_session_is_noop_and_active_hook_is_not(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    sess = XAISession(capabilities["moss"], interventions=[InterventionSpec("logit_knockout", params=_KO)])
    native = native_sdpa()
    with sess.disabled():
        out = sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native)
        assert sess.interventions_for("logit_knockout", 0, make_ctx()) == []
    assert native.calls == 1 and maxdiff(out, native(q, k, v)) == 0
    native = native_sdpa()
    hooked = sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native)
    assert native.calls == 0 and maxdiff(hooked, native(q, k, v)) > 1e-3


def test_layout_mismatch_raises(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    q, k, v = make_qkv(layouts["img_txt"])
    sess = XAISession(capabilities["moss"], interventions=[InterventionSpec("logit_knockout", params=_KO)])
    with pytest.raises(ValueError):
        sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=SeqLayout("img_txt", 16, 6), native=native_sdpa())


# =========================================================================== fixed CFG batching


def test_fixed_cfg_batching_guard(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["img_txt"]
    q4, k4, v4 = make_qkv(layout, batch=4)
    q2, k2, v2 = make_qkv(layout, batch=2)
    sess = XAISession(capabilities["moss"])
    native = native_sdpa()
    sess.attention(q4, k4, v4, block=0, ctx=make_ctx((COND, UNCOND), chunk=2), layout=layout, native=native)
    # branch-order swap keeps the policy (2 chunks x 2 rows)
    sess.attention(q4, k4, v4, block=1, ctx=make_ctx((UNCOND, COND), chunk=2, sigma=0.875), layout=layout,
                   native=native)
    with pytest.raises(RuntimeError, match="batching"):
        sess.attention(q2, k2, v2, block=0, ctx=make_ctx((COND,), chunk=2, sigma=0.75), layout=layout, native=native)

    lax = XAISession(capabilities["moss"], enforce_fixed_batching=False)
    lax.attention(q4, k4, v4, block=0, ctx=make_ctx((COND, UNCOND), chunk=2), layout=layout, native=native)
    lax.attention(q2, k2, v2, block=0, ctx=make_ctx((COND,), chunk=2), layout=layout, native=native)


def test_split_policy_is_also_fixed(capabilities, make_ctx):
    sess = XAISession(capabilities["moss"])
    sess.check_batching(make_ctx((COND,), chunk=2))
    sess.check_batching(make_ctx((UNCOND,), chunk=2))
    with pytest.raises(RuntimeError):
        sess.check_batching(make_ctx((COND, UNCOND), chunk=2))


# =========================================================================== capture mapping


def test_capture_follows_cond_rows_under_branch_swap(capabilities, layouts, make_qkv, make_ctx, dense_attention,
                                                     native_sdpa):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout, batch=4)
    ctx = make_ctx((UNCOND, COND), chunk=2)
    sess = XAISession(capabilities["moss"], capture=CaptureSpec(fields={"q", "probs"}), query_chunk=5, key_chunk=3)
    native = native_sdpa()
    out = sess.attention(q, k, v, block=0, ctx=ctx, layout=layout, native=native)
    assert maxdiff(out, native(q, k, v)) < 1e-5
    (key, entry), = sess.captures.items()
    assert key == (0, 0, (("cond", 0), ("cond", 1)))
    assert entry.rows == (2, 3) and entry.present() == ("q", "probs")
    assert torch.equal(entry.q, q[2:4])
    p, _ = dense_attention(q, k)
    assert maxdiff(entry.probs, p[2:4]) < 1e-5


# =========================================================================== RunningMap


def test_running_map_weighted_mean():
    rm = RunningMap("mean")
    a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0])
    rm.add(a, weight=1.0)
    rm.add(b, weight=3.0)
    expected = (a + 3 * b) / 4
    assert maxdiff(rm.value(), expected) < 1e-6
    assert maxdiff(rm.value(), expected) < 1e-6  # non-destructive
    assert rm.count == 2 and rm.shape == (2,)


def test_running_map_group_mean_sum():
    rm = RunningMap("group_mean_sum")
    a, b, c = torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([10.0])
    rm.add(a, group=0)
    rm.add(b, group=0)
    assert maxdiff(rm.value(), torch.tensor([2.0])) < 1e-6
    rm.add(c, group=1)
    assert maxdiff(rm.value(), torch.tensor([12.0])) < 1e-6
    with pytest.raises(ValueError, match="contiguous"):
        rm.add(a, group=0)
    with pytest.raises(ValueError):
        rm.add(a)  # group required


def test_running_map_errors():
    with pytest.raises(ValueError):
        RunningMap("median")
    rm = RunningMap()
    with pytest.raises(ValueError):
        rm.value()
    rm.add(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="shape"):
        rm.add(torch.zeros(3, 2))


# =========================================================================== small helpers


def test_index_and_slice_helpers():
    assert index_mask(5, None) is None
    assert index_mask(5, slice(1, 3)).tolist() == [False, True, True, False, False]
    assert index_mask(5, [0, 4]).tolist() == [True, False, False, False, True]
    with pytest.raises(ValueError):
        index_mask(5, torch.ones(4, dtype=torch.bool))
    assert union_mask(6, [slice(0, 1), [4]]).tolist() == [True, False, False, False, True, False]
    assert slice_overlap(slice(5, 10), slice(8, 20)) == (slice(3, 5), slice(0, 2))
    assert slice_overlap(slice(0, 4), slice(4, 8)) is None
