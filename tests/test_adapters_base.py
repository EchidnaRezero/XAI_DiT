"""Adapter contract invariants (``plan/m1.md`` §3: Adapter / CFG, TokenMap).

Pure Python: ``xai_dit.adapters.base`` does not import torch, so this module runs on a
CPU-only box without PyTorch (one tensor-input case skips itself).
"""

from __future__ import annotations

import random

import pytest

from xai_dit.adapters.base import (
    _REQUIRED_PARAMS,
    CAPTURE_FIELDS,
    COND,
    INTERVENTION_HOOKS,
    UNCOND,
    AdapterCapability,
    CaptureSpec,
    InterventionSpec,
    SeqLayout,
    StepContext,
    StepWindow,
    TokenAlignmentError,
    TokenMap,
    TokenSpan,
    as_spans,
)

SIGMAS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0)


def _ctx(order=(COND, UNCOND), chunk=1, sigma=1.0) -> StepContext:
    return StepContext(SIGMAS, sigma, tuple(order), chunk * len(order))


def _row(n: int, rng: random.Random, total: float = 1.0) -> list[float]:
    w = [rng.random() + 1e-3 for _ in range(n)]
    s = sum(w)
    return [total * x / s for x in w]


# =========================================================================== M1 contract lists


def test_capture_fields_and_hooks_match_m1_scope():
    assert set(CAPTURE_FIELDS) == {"q", "k", "v", "probs", "attn_pre_proj", "attn_out", "gate_logits",
                                   "gate_value", "residual_out"}
    assert set(INTERVENTION_HOOKS) == {"kv_patch", "logit_knockout", "logit_scale", "attn_override",
                                       "two_pass_sink_mask", "span_query_mask", "pre_dit_span_swap",
                                       "gate_override", "per_head_token_ablation", "concept_stream",
                                       "true_block_bypass"}
    assert set(_REQUIRED_PARAMS) == set(INTERVENTION_HOOKS)


# =========================================================================== TokenSpan


def test_token_span_is_half_open():
    s = TokenSpan(2, 5)
    assert len(s) == 3
    assert 2 in s and 4 in s
    assert 5 not in s and 1 not in s
    assert "2" not in s and 2.0 not in s  # only ints are positions
    assert list(s.indices()) == [2, 3, 4]
    assert s.as_slice() == slice(2, 5)
    assert s.as_slice(16) == slice(18, 21)
    assert len(TokenSpan(3, 3)) == 0


def test_token_span_adjacent_spans_do_not_overlap():
    assert not TokenSpan(0, 3).overlaps(TokenSpan(3, 5))
    assert TokenSpan(0, 4).overlaps(TokenSpan(3, 5))
    assert not TokenSpan(2, 2).overlaps(TokenSpan(0, 5))  # empty span overlaps nothing


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2)])
def test_token_span_rejects_invalid(start, end):
    with pytest.raises(ValueError):
        TokenSpan(start, end)


def test_as_spans():
    s = TokenSpan(0, 1)
    assert as_spans(s) == (s,)
    assert as_spans([s, TokenSpan(1, 2)]) == (s, TokenSpan(1, 2))


# =========================================================================== TokenMap partition


def _anima_map(L: int = 12, Lt: int = 7, **kw) -> TokenMap:
    """Anima T5 slots: ``S=[0, Lt-1)``, ``R=[Lt-1, Lt)`` (``</s>``), ``P=[Lt, L)`` (zero pad)."""
    ids = list(range(100, 100 + Lt - 1)) + [1] + [0] * (L - Lt)
    return TokenMap.from_lengths(ids, suffix=1, pad=L - Lt, **kw)


def test_anima_token_map_spans():
    tm = _anima_map(L=12, Lt=7)
    assert dict(tm.spans) == {"S": TokenSpan(0, 6), "R": TokenSpan(6, 7), "P": TokenSpan(7, 12)}
    assert [tm.role_of(i) for i in range(12)] == ["S"] * 6 + ["R"] + ["P"] * 5


def test_anima_partition_sum_equals_one():
    """Cross-attn row over all L text slots: ``S + R + P == 1`` (``max_abs_diff < 1e-6``)."""
    rng = random.Random(0)
    tm = _anima_map(L=12, Lt=7)
    for _ in range(20):
        row = _row(tm.text_len, rng)
        parts = {n: sum(row[i] for i in sp.indices()) for n, sp in tm.spans.items()}
        assert abs(sum(parts.values()) - 1.0) < 1e-6
        # each slot counted exactly once
        assert sorted(i for sp in tm.spans.values() for i in sp.indices()) == list(range(tm.text_len))


def test_joint_partition_sum_equals_text_share():
    """Joint-attn (MOSS / Krea 2): ``R_prefix + S + R == sum_txt`` (< 1, image keys take the rest)."""
    rng = random.Random(1)
    N, L = 16, 9
    tm = TokenMap.from_lengths(list(range(L)), prefix=2, suffix=3)
    assert dict(tm.spans) == {"R_prefix": TokenSpan(0, 2), "S": TokenSpan(2, 6), "R": TokenSpan(6, 9)}
    for order in ("img_txt", "txt_img"):
        lay = SeqLayout(order, N, L)
        row = _row(N + L, rng)
        txt = row[lay.k_txt]
        sum_txt = sum(txt)
        parts = sum(sum(row[lay.txt_keys(sp)]) for sp in tm.spans.values())
        assert abs(parts - sum_txt) < 1e-6
        assert sum_txt < 1.0


def test_krea2_suffix_register_span():
    L = 13
    tm = TokenMap.from_lengths(list(range(L)), suffix=5)
    assert tm.spans["S"] == TokenSpan(0, L - 5) and tm.spans["R"] == TokenSpan(L - 5, L)


@pytest.mark.parametrize("spans", [
    {"S": TokenSpan(0, 3), "R": TokenSpan(4, 6)},                       # gap
    {"S": TokenSpan(0, 4), "R": TokenSpan(3, 6)},                       # overlap
    {"S": TokenSpan(0, 5)},                                             # short coverage
    {"S": TokenSpan(0, 3), "E": TokenSpan(3, 3), "R": TokenSpan(3, 6)},  # empty span
    {"S": TokenSpan(1, 6)},                                             # does not start at 0
])
def test_token_map_rejects_non_partitions(spans):
    with pytest.raises(ValueError):
        TokenMap(tuple(range(6)), spans)


def test_token_map_rejects_targets_beyond_text():
    with pytest.raises(ValueError):
        TokenMap(tuple(range(4)), {"S": TokenSpan(0, 4)}, targets={"t": TokenSpan(2, 5)})


def test_from_lengths_rejects_overflow():
    with pytest.raises(ValueError):
        TokenMap.from_lengths([1, 2, 3], prefix=2, suffix=2)


def test_occurrence_index():
    tm = TokenMap((5, 7, 5, 5), {"S": TokenSpan(0, 4)})
    assert tm.occurrence_idx == (0, 0, 1, 2)
    assert tm.occurrence(5, 0) == 0 and tm.occurrence(5, 2) == 3 and tm.occurrence(7) == 1
    with pytest.raises(KeyError):
        tm.occurrence(5, 3)
    with pytest.raises(ValueError):
        TokenMap((5, 7), {"S": TokenSpan(0, 2)}, occurrence_idx=(0,))


# =========================================================================== strict token alignment


def _pair_maps(base_ids, other_ids, **kw):
    t = {"tag": TokenSpan(2, 4)}
    return (TokenMap.from_lengths(base_ids, suffix=1, pad=1, targets=t, **kw),
            TokenMap.from_lengths(other_ids, suffix=1, pad=1, targets=t, **kw))


def test_strict_alignment_accepts_edits_inside_editable_span():
    a, b = _pair_maps([11, 12, 13, 14, 15, 16, 2, 0], [11, 12, 93, 94, 15, 16, 2, 0])
    a.assert_strict_token_alignment(b, editable=["tag"])
    a.assert_strict_token_alignment(b, editable=[TokenSpan(2, 4)])
    a.assert_strict_token_alignment(a)  # identity


def test_strict_alignment_rejects_edit_outside_span():
    a, b = _pair_maps([11, 12, 13, 14, 15, 16, 2, 0], [11, 12, 93, 94, 99, 16, 2, 0])
    with pytest.raises(TokenAlignmentError, match=r"\[4\]"):
        a.assert_strict_token_alignment(b, editable=["tag"])
    # boundary is half-open: position 4 == span.end is *outside* [2, 4)
    with pytest.raises(TokenAlignmentError):
        a.assert_strict_token_alignment(b, editable=[TokenSpan(2, 4)])
    assert issubclass(TokenAlignmentError, AssertionError)


def test_strict_alignment_rejects_length_and_partition_mismatch():
    a = TokenMap.from_lengths([1, 2, 3, 4, 5, 0], suffix=1, pad=1)
    shorter = TokenMap.from_lengths([1, 2, 3, 4, 0], suffix=1, pad=1)
    with pytest.raises(TokenAlignmentError, match="length"):
        a.assert_strict_token_alignment(shorter)
    other_partition = TokenMap.from_lengths([1, 2, 3, 4, 5, 0], suffix=2, pad=0)
    with pytest.raises(TokenAlignmentError, match="partition"):
        a.assert_strict_token_alignment(other_partition)


# =========================================================================== capability / layout


def test_capability_gqa_indexing():
    cap = AdapterCapability(name="x", attention_kind="joint_attn", seq_order="img_txt", num_blocks=2,
                            num_q_heads=8, num_kv_heads=2, head_dim=16, text_updated_per_block=True,
                            supports_concept_attention=True, has_output_gate=False)
    assert cap.gqa_group == 4
    assert [cap.kv_head(h) for h in range(8)] == [h // 4 for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(cap.q_heads_of(1)) == [4, 5, 6, 7]


@pytest.mark.parametrize("kw", [
    dict(attention_kind="cross_attn", seq_order="img_txt"),
    dict(attention_kind="joint_attn", seq_order="cross"),
    dict(num_kv_heads=3),
    dict(head_dim=0),
])
def test_capability_validation(kw):
    base = dict(name="x", attention_kind="joint_attn", seq_order="img_txt", num_blocks=2, num_q_heads=8,
                num_kv_heads=2, head_dim=16, text_updated_per_block=True, supports_concept_attention=True,
                has_output_gate=False)
    base.update(kw)
    with pytest.raises(ValueError):
        AdapterCapability(**base)


def test_seq_layout_slices():
    it, ti, cr = SeqLayout("img_txt", 16, 8), SeqLayout("txt_img", 16, 8), SeqLayout("cross", 16, 8)
    assert (it.q_img, it.q_txt, it.k_img, it.k_txt) == (slice(0, 16), slice(16, 24), slice(0, 16), slice(16, 24))
    assert (ti.q_img, ti.q_txt, ti.k_img, ti.k_txt) == (slice(8, 24), slice(0, 8), slice(8, 24), slice(0, 8))
    assert (cr.n_query, cr.n_key, cr.q_img, cr.q_txt, cr.k_img, cr.k_txt) == (16, 8, slice(0, 16), None, None, slice(0, 8))
    assert it.txt_keys(TokenSpan(2, 4)) == slice(18, 20)
    assert ti.txt_keys(TokenSpan(2, 4)) == slice(2, 4)
    assert ti.txt_queries(TokenSpan(2, 4)) == slice(2, 4)
    with pytest.raises(ValueError):
        cr.txt_queries(TokenSpan(0, 1))
    with pytest.raises(ValueError):
        it.txt_keys(TokenSpan(7, 9))


# =========================================================================== StepContext / CFG


@pytest.mark.parametrize("sigma,step", [(1.0, 0), (0.875, 1), (0.5, 4), (0.125, 7), (0.5 + 1e-7, 4)])
def test_step_index_derived_from_sigma(sigma, step):
    ctx = _ctx(sigma=sigma)
    assert ctx.step_index == step
    assert ctx.num_steps == 8
    assert ctx.step_fraction == pytest.approx(step / 8)


def test_step_index_is_not_call_counting():
    """Repeated forwards at the same sigma (CFG split calls, re-runs) map to the same step."""
    assert {_ctx(order=o, sigma=0.625).step_index for o in [(COND, UNCOND), (COND,), (UNCOND,)]} == {3}


def test_step_index_from_transformer_options():
    opts = {"sigmas": [0.75, 0.75], "sample_sigmas": list(SIGMAS), "cond_or_uncond": [1, 0]}
    ctx = StepContext.from_transformer_options(opts, batch_size=4)
    assert ctx.step_index == 2 and ctx.cond_or_uncond == (UNCOND, COND) and ctx.chunk_size == 2
    with pytest.raises(ValueError, match="mixed"):
        StepContext.from_transformer_options({**opts, "sigmas": [0.75, 0.5]}, batch_size=4)
    with pytest.raises(ValueError, match="not found"):
        StepContext.from_transformer_options({**opts, "sigmas": [0.7, 0.7]}, batch_size=4)


def test_step_index_from_tensor_options():
    torch = pytest.importorskip("torch")
    opts = {"sigmas": torch.tensor([0.25, 0.25]), "sample_sigmas": torch.tensor(SIGMAS),
            "cond_or_uncond": [0, 1]}
    assert StepContext.from_transformer_options(opts, batch_size=2).step_index == 6


@pytest.mark.parametrize("order,batch", [((COND, UNCOND), 3), ((), 2), ((COND, 2), 2)])
def test_step_context_validation(order, batch):
    with pytest.raises(ValueError):
        StepContext(SIGMAS, 1.0, order, batch)
    with pytest.raises(ValueError):
        StepContext((), 1.0, (COND,), 1)


def test_cond_or_uncond_branch_order_swap():
    """Per-sample data laid out by ``row_map`` is read back identically for both chunk orders."""
    data = {("cond", 0): "c0", ("cond", 1): "c1", ("uncond", 0): "u0", ("uncond", 1): "u1"}
    a, b = _ctx((COND, UNCOND), chunk=2), _ctx((UNCOND, COND), chunk=2)
    assert a.row_map() == [("cond", 0), ("cond", 1), ("uncond", 0), ("uncond", 1)]
    assert b.row_map() == [("uncond", 0), ("uncond", 1), ("cond", 0), ("cond", 1)]
    assert a.rows("cond") == (0, 1) and b.rows("cond") == (2, 3)
    for ctx in (a, b):
        batch = [data[rm] for rm in ctx.row_map()]
        assert [batch[r] for r in ctx.rows("cond")] == ["c0", "c1"]
        assert [batch[r] for r in ctx.rows("uncond")] == ["u0", "u1"]
    assert sorted(a.row_map()) == sorted(b.row_map())


def test_cond_or_uncond_split_calls_match_batched():
    batched = _ctx((COND, UNCOND), chunk=2)
    c, u = _ctx((COND,), chunk=2), _ctx((UNCOND,), chunk=2)
    assert c.row_map() + u.row_map() == batched.row_map()
    assert c.rows("uncond") == () and u.rows("cond") == ()
    assert c.rows("cond") == (0, 1) and u.rows("uncond") == (0, 1)
    assert c.rows_for_scope("cond_only") == (0, 1) and u.rows_for_scope("cond_only") == ()
    assert u.rows_for_scope("both") == (0, 1)


def test_cond_or_uncond_b_greater_than_one():
    ctx = _ctx((UNCOND, COND), chunk=3)
    assert ctx.batch_size == 6 and ctx.chunk_size == 3
    assert ctx.rows("uncond") == (0, 1, 2) and ctx.rows("cond") == (3, 4, 5)
    assert [sid for b, sid in ctx.row_map() if b == "cond"] == [0, 1, 2]


def test_step_window():
    w = StepWindow(start=2, end=4)
    assert [w.contains(_ctx(sigma=s)) for s in SIGMAS[:-1]] == [False, False, True, True, False, False, False, False]
    sw = StepWindow(sigma_min=0.97)
    assert sw.contains(_ctx(sigma=1.0)) and not sw.contains(_ctx(sigma=0.875))
    assert StepWindow(sigma_max=0.5).contains(_ctx(sigma=0.5))  # inclusive sigma bounds


# =========================================================================== capture / intervention specs


def test_capture_spec_validation_and_rows():
    spec = CaptureSpec(fields={"q", "probs"}, blocks=[1, 2], branches={"cond", "uncond"})
    assert spec.fields == frozenset({"q", "probs"}) and spec.blocks == frozenset({1, 2})
    ctx = _ctx((UNCOND, COND), chunk=2)
    assert spec.rows(ctx) == (2, 3, 0, 1)  # cond rows first, then uncond
    assert spec.wants("q", 1, ctx) and not spec.wants("q", 0, ctx) and not spec.wants("v", 1, ctx)
    assert CaptureSpec(fields={"q"}).rows(ctx) == (2, 3)  # default branches = cond only
    with pytest.raises(ValueError):
        CaptureSpec(fields={"logits"})
    with pytest.raises(ValueError):
        CaptureSpec(fields={"q"}, branches={"both"})


@pytest.mark.parametrize("hook", INTERVENTION_HOOKS)
def test_intervention_spec_required_params(hook):
    required = _REQUIRED_PARAMS[hook]
    spec = InterventionSpec(hook, params={p: None for p in required})
    assert spec.hook == hook
    if required:
        with pytest.raises(ValueError, match="requires params"):
            InterventionSpec(hook, params={})


def test_intervention_spec_validation_and_scope():
    with pytest.raises(ValueError, match="unknown hook"):
        InterventionSpec("attention_rollout")
    with pytest.raises(ValueError, match="cfg_scope"):
        InterventionSpec("true_block_bypass", cfg_scope="uncond_only")
    spec = InterventionSpec("logit_knockout", blocks=[0, 2], window=StepWindow(end=2), cfg_scope="cond_only",
                            params={"keys": TokenSpan(0, 1)})
    assert spec.blocks == frozenset({0, 2})
    assert spec.active(2, _ctx(sigma=0.875)) and not spec.active(1, _ctx()) and not spec.active(0, _ctx(sigma=0.75))
    ctx = _ctx((UNCOND, COND), chunk=2)
    assert spec.rows(ctx) == (2, 3)
    both = InterventionSpec("true_block_bypass")
    assert both.rows(ctx) == (0, 1, 2, 3) and both.active(99, ctx)
