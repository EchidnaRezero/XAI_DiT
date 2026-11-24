"""Direction 03 patching invariants (``plan/m1.md`` §3 E3; ``docs/specs/03_tag_influence.md``).

E3-0 Same-State ``direct_effect`` vs ``total_effect`` and the rollout bridge; E3-1 K/V cache;
E3-2 PCI Euler prefix-cache rollout == full rollout (``< 1e-5``), ``tau`` crossings,
reference-normalised score; E3-3 TACA sign monotonicity and Listing 2 text-query restoration;
E3-4 DiT-ST normalised boundary and SAP stage schedule; offline BF16 cond cache.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed; run on a torch host")
import torch.nn.functional as F  # noqa: E402

from xai_dit.adapters.base import (  # noqa: E402
    COND,
    UNCOND,
    SeqLayout,
    StepWindow,
    TokenAlignmentError,
    TokenMap,
    TokenSpan,
)
from xai_dit.core.patching import (  # noqa: E402
    AttentionChangeTracker,
    CondBundle,
    CondCache,
    KVCache,
    PCIController,
    StagePromptSchedule,
    attention_change,
    attribute_boundary,
    cfg_combine,
    check_total_effect_pair,
    conditioning_sequence,
    direct_effect_spec,
    discrete_curvature,
    euler_step,
    kv_group_heads,
    moving_average,
    normalize_by_dsigma,
    random_token_control,
    reference_normalized_score,
    relation_boundary,
    rollout_bridge,
    same_state_effect,
    snr,
    switch_points,
    taca_gamma,
    taca_listing2_attention,
    taca_spec,
    tau_crossings,
)
from xai_dit.core.session import (  # noqa: E402
    LogitEdit,
    Region,
    XAISession,
    chunked_attention,
    expand_kv,
    index_mask,
    union_mask,
)

SIGMAS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0)


def maxdiff(a, b) -> float:
    return (a.double() - b.double()).abs().max().item()


# =========================================================================== E3-0 Same-State effects


def test_direct_effect_knocks_out_img_to_span_only(capabilities, layouts, make_qkv, make_ctx, prob_collector,
                                                   native_sdpa):
    layout = layouts["img_txt"]
    span = TokenSpan(2, 4)
    q, k, v = make_qkv(layout)
    spec = direct_effect_spec(span)
    assert spec.hook == "logit_knockout" and spec.params["query"] == "img"
    sess = XAISession(capabilities["moss"], interventions=[spec])
    store: list = []
    sess.add_observer(prob_collector.factory(store))
    out = sess.attention(q, k, v, block=0, ctx=make_ctx(), layout=layout, native=native_sdpa())
    P = store[0].probs
    ks = layout.txt_keys(span)
    assert P[:, :, layout.q_img, ks].abs().max().item() == 0
    assert maxdiff(P.sum(-1), torch.ones(P.shape[:3])) < 1e-6

    # == attention with the span keys removed for image queries (direct path only)
    keep = [j for j in range(layout.n_key) if not ks.start <= j < ks.stop]
    ref = F.scaled_dot_product_attention(q[:, :, layout.q_img], expand_kv(k[:, :, keep], 4), expand_kv(v[:, :, keep], 4))
    assert maxdiff(out[:, :, layout.q_img], ref) < 1e-5
    base = chunked_attention(q, k, v)
    assert maxdiff(out[:, :, layout.q_txt], base[:, :, layout.q_txt]) == 0  # text queries untouched


def _toy_velocity(x, token_ids, *, knockout=None, seed=0):
    """One cross-attn velocity head at a *fixed* state ``x`` [B, N, D]; text K/V are token-wise projections."""
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(128, 16, generator=g)
    wk, wv = torch.randn(16, 16, generator=g) * 0.3, torch.randn(16, 16, generator=g)
    txt = emb[torch.tensor(token_ids)]
    k_txt, v_txt = (txt @ wk).view(1, len(token_ids), 2, 8), (txt @ wv).view(1, len(token_ids), 2, 8)
    B, N, _ = x.shape
    q = x.view(B, N, 2, 8).transpose(1, 2)
    k = k_txt.transpose(1, 2).expand(B, -1, -1, -1)
    v = v_txt.transpose(1, 2).expand(B, -1, -1, -1)
    edits = []
    if knockout is not None:
        edits.append(LogitEdit(Region(key=index_mask(len(token_ids), knockout.as_slice())), "set", float("-inf")))
    out = chunked_attention(q, k, v, logit_edits=edits)
    return out.transpose(1, 2).reshape(B, N, 16), k, v


def test_total_effect_same_state_isolated_to_span():
    base_ids = [11, 12, 13, 14, 15, 16, 2, 0]
    neutral_ids = [11, 12, 93, 94, 15, 16, 2, 0]
    targets = {"tag": TokenSpan(2, 4)}
    tm_b = TokenMap.from_lengths(base_ids, suffix=1, pad=1, targets=targets)
    tm_n = TokenMap.from_lengths(neutral_ids, suffix=1, pad=1, targets=targets)
    check_total_effect_pair(tm_b, tm_n, "tag")
    with pytest.raises(TokenAlignmentError):
        check_total_effect_pair(tm_b, TokenMap.from_lengths([11, 99, 93, 94, 15, 16, 2, 0], suffix=1, pad=1,
                                                            targets=targets), "tag")

    x = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(1))  # the same state x_sigma for all arms
    v_base, k_b, vv_b = _toy_velocity(x, base_ids)
    v_tot, k_n, vv_n = _toy_velocity(x, neutral_ids)
    v_dir, _, _ = _toy_velocity(x, base_ids, knockout=TokenSpan(2, 4))
    outside = [0, 1, 4, 5, 6, 7]
    assert torch.equal(k_b[:, :, outside], k_n[:, :, outside]) and torch.equal(vv_b[:, :, outside], vv_n[:, :, outside])
    assert not torch.equal(k_b[:, :, 2:4], k_n[:, :, 2:4])

    sigma, sigma_next = 0.75, 0.625
    tot = same_state_effect(v_base, v_tot, sigma, sigma_next, channel_dim=-1)
    dire = same_state_effect(v_base, v_dir, sigma, sigma_next, channel_dim=-1)
    assert tot["raw"].shape == (2, 16)
    assert maxdiff(tot["weighted"], 0.125 * tot["raw"]) < 1e-7
    assert tot["raw"].max() > 1e-4 and dire["raw"].max() > 1e-4
    assert maxdiff(tot["raw"], dire["raw"]) > 1e-4  # knockout != substitution
    zero = same_state_effect(v_base, _toy_velocity(x, base_ids)[0], sigma, sigma_next, channel_dim=-1)
    assert zero["raw"].abs().max().item() == 0


def test_random_token_control():
    tm = TokenMap.from_lengths(list(range(10)), suffix=1, pad=1)  # S = [0, 8)
    ids = random_token_control(tm, 3, exclude=[TokenSpan(2, 4)], generator=torch.Generator().manual_seed(0))
    assert len(ids) == 3 and ids == sorted(ids)
    assert all(0 <= i < 8 and i not in (2, 3) for i in ids)
    with pytest.raises(ValueError):
        random_token_control(tm, 7, exclude=[TokenSpan(2, 4)])


def test_rollout_bridge_and_cfg_combine():
    g = torch.Generator().manual_seed(2)
    delta_v = [torch.randn(1, 4, 8, 8, generator=g) for _ in range(3)]
    sigmas = [1.0, 0.75, 0.5, 0.25]
    pred = sum((sigmas[i + 1] - sigmas[i]) * dv for i, dv in enumerate(delta_v))
    r = rollout_bridge(delta_v, sigmas, 2 * pred)
    assert r["bridge_corr"] == pytest.approx(1.0, abs=1e-5) and r["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert rollout_bridge(delta_v, sigmas, -pred)["cosine"] == pytest.approx(-1.0, abs=1e-5)
    with pytest.raises(ValueError):
        rollout_bridge(delta_v, sigmas[:3], pred)
    vc, vu = torch.randn(2, 4, generator=g), torch.randn(2, 4, generator=g)
    assert maxdiff(cfg_combine(vc, vu, 1.0), vc) < 1e-6
    assert maxdiff(cfg_combine(vc, vu, 0.0), vu) == 0
    assert maxdiff(cfg_combine(vc, vu, 4.5), vu + 4.5 * (vc - vu)) == 0


# =========================================================================== E3-1 K/V cache


def _kv(layout, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, 2, layout.n_key, 16, generator=g), torch.randn(batch, 2, layout.n_key, 16, generator=g))


def test_kv_cache_patches_text_slice_branch_to_branch(layouts, make_ctx):
    layout = layouts["img_txt"]
    s = layout.k_txt
    ks, vs = _kv(layout, 4, 0)
    kt, vt = _kv(layout, 4, 1)
    cache = KVCache("txt")
    cache.store(0, make_ctx((COND, UNCOND), chunk=2), ks, vs, layout)

    k2, v2 = cache.patch(0, make_ctx((COND, UNCOND), chunk=2), kt, vt, layout)
    assert torch.equal(k2[:, :, s], ks[:, :, s]) and torch.equal(v2[:, :, s], vs[:, :, s])
    assert torch.equal(k2[:, :, layout.k_img], kt[:, :, layout.k_img])
    assert not torch.equal(kt[:, :, s], ks[:, :, s])  # the target itself was not modified in place

    # branch-order swap: target row 0 is uncond sample 0 == source row 2
    k3, _ = cache.patch(0, make_ctx((UNCOND, COND), chunk=2), kt, vt, layout)
    for dst, src in [(0, 2), (1, 3), (2, 0), (3, 1)]:
        assert torch.equal(k3[dst, :, s], ks[src, :, s])

    # split CFG calls are equivalent to the batched call
    kc, _ = cache.patch(0, make_ctx((COND,), chunk=2), kt[:2], vt[:2], layout)
    ku, _ = cache.patch(0, make_ctx((UNCOND,), chunk=2), kt[2:], vt[2:], layout)
    assert torch.equal(torch.cat([kc, ku])[:, :, s], ks[:, :, s])

    # cond_only scope leaves uncond rows untouched
    k4, _ = cache.patch(0, make_ctx((COND, UNCOND), chunk=2), kt, vt, layout, cfg_scope="cond_only")
    assert torch.equal(k4[:2, :, s], ks[:2, :, s]) and torch.equal(k4[2:], kt[2:])


def test_kv_cache_heads_and_spans(capabilities, layouts, make_ctx):
    layout = layouts["img_txt"]
    s = layout.k_txt
    ctx = make_ctx()
    ks, vs = _kv(layout, 2, 0)
    kt, vt = _kv(layout, 2, 1)
    cache = KVCache("txt")
    cache.store(0, ctx, ks, vs, layout)

    k1, v1 = cache.patch(0, ctx, kt, vt, layout, kv_heads=[1])
    assert torch.equal(k1[:, 0], kt[:, 0]) and torch.equal(k1[:, 1, s], ks[:, 1, s])
    # only the Q heads that read KV head 1 change
    q = torch.randn(2, 8, layout.n_query, 16, generator=torch.Generator().manual_seed(2))
    o_t, o_p = chunked_attention(q, kt, vt), chunked_attention(q, k1, v1)
    group0, group1 = list(kv_group_heads(0, 4)), list(kv_group_heads(1, 4))
    assert group1 == list(capabilities["moss"].q_heads_of(1))
    assert maxdiff(o_t[:, group0], o_p[:, group0]) == 0
    assert maxdiff(o_t[:, group1], o_p[:, group1]) > 1e-4

    k5, _ = cache.patch(0, ctx, kt, vt, layout, spans=[TokenSpan(1, 3)])
    lo = s.start
    assert torch.equal(k5[:, :, lo + 1:lo + 3], ks[:, :, lo + 1:lo + 3])
    assert torch.equal(k5[:, :, lo:lo + 1], kt[:, :, lo:lo + 1]) and torch.equal(k5[:, :, lo + 3:], kt[:, :, lo + 3:])


def test_kv_cache_errors(layouts, make_ctx):
    layout = layouts["img_txt"]
    ks, vs = _kv(layout, 2, 0)
    cache = KVCache("txt")
    cache.store(0, make_ctx(), ks, vs, layout)
    with pytest.raises(KeyError):
        cache.patch(0, make_ctx(sigma=0.5), ks, vs, layout)  # no cache for this step
    short = SeqLayout("img_txt", 16, 6)
    k6, v6 = _kv(short, 2, 1)
    with pytest.raises(ValueError, match="length"):
        cache.patch(0, make_ctx(), k6, v6, short)
    with pytest.raises(ValueError):
        KVCache("img").store(0, make_ctx(), *_kv(layouts["cross"], 2, 0), layouts["cross"])
    img = KVCache("img")
    img.store(0, make_ctx(), ks, vs, layout)
    kp, _ = img.patch(0, make_ctx(), *_kv(layout, 2, 3), layout)
    assert torch.equal(kp[:, :, layout.k_img], ks[:, :, layout.k_img])


# =========================================================================== E3-2 PCI


def _pci_denoise(log):
    W = torch.randn(16, 16, generator=torch.Generator().manual_seed(11)) * 0.3

    def denoise(x, sigma, i, cond):
        label, c = cond
        log.append((i, label))
        return torch.tanh(x @ W + c * sigma)

    return denoise


def test_pci_prefix_cache_matches_full_rollout():
    log: list = []
    g = torch.Generator().manual_seed(12)
    x_T = torch.randn(2, 16, generator=g)
    cb, cc = ("base", torch.randn(16, generator=g)), ("concept", torch.randn(16, generator=g))
    pci = PCIController(_pci_denoise(log), SIGMAS)
    S = pci.num_steps
    states = pci.cache_base(x_T, cb)
    assert S == 8 and len(states) == S + 1 and states[0] is x_T
    assert switch_points(S, stride=3) == [0, 3, 6, 8]
    for k in switch_points(S):
        before = pci.calls
        cached = pci.switch(k, cc)
        assert pci.calls - before == S - k  # O(S - k) instead of O(S)
        log.clear()
        full = pci.full_rollout(x_T, k, cb, cc)
        assert [label for _, label in log] == conditioning_sequence(k, S)
        assert maxdiff(cached, full) < 1e-5
    assert maxdiff(pci.switch(S, cc), states[-1]) == 0  # k = S is pure base
    log.clear()
    pure_concept = pci.full_rollout(x_T, 0, cb, cc)
    assert all(label == "concept" for _, label in log)
    assert maxdiff(pci.switch(0, cc), pure_concept) < 1e-5


def test_pci_errors_and_euler_step():
    pci = PCIController(lambda x, s, i, c: x, SIGMAS)
    with pytest.raises(RuntimeError):
        pci.switch(0, None)
    pci.cache_base(torch.zeros(1), None)
    with pytest.raises(ValueError):
        pci.switch(9, None)
    with pytest.raises(ValueError):
        PCIController(lambda *a: a[0], [1.0])
    x, v = torch.tensor([1.0]), torch.tensor([2.0])
    assert euler_step(x, v, 1.0, 0.75).item() == pytest.approx(0.5)


def test_tau_crossings():
    tau = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = tau_crossings(tau, [0.1, 0.3, 0.6, 0.8, 0.9])
    assert (r["tau_50"], r["status_50"], r["tau_70"], r["status_70"]) == (0.5, "ok", 0.75, "ok")
    assert r["W"] == pytest.approx(0.25)
    shuffled = tau_crossings(tau[::-1], [0.9, 0.8, 0.6, 0.3, 0.1])  # unsorted input is sorted by tau
    assert shuffled["tau_50"] == 0.5
    never = tau_crossings(tau, [0.1, 0.2, 0.3, 0.4, 0.45])
    assert math.isnan(never["tau_50"]) and never["status_50"] == "never" and math.isnan(never["W"])
    always = tau_crossings(tau, [0.55, 0.6, 0.65, 0.8, 0.9])
    assert math.isnan(always["tau_50"]) and always["status_50"] == "always"
    assert always["status_70"] == "ok" and always["tau_70"] == 0.75


def test_reference_normalized_score():
    s = reference_normalized_score(torch.tensor([0.2, 0.8, 0.5]), 0.2, 0.8)
    assert torch.allclose(s, torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64))
    assert math.isnan(float(reference_normalized_score(0.3, 0.5, 0.5)))


# =========================================================================== E3-3 TACA


def test_taca_gamma_and_window(make_ctx):
    assert taca_gamma(0.98) == 1.2 and taca_gamma(0.97) == 1.2 and taca_gamma(0.96) == 1.0
    spec = taca_spec()
    assert spec.hook == "logit_scale" and spec.window == StepWindow(sigma_min=0.97)
    assert spec.active(0, make_ctx(sigma=1.0)) and not spec.active(0, make_ctx(sigma=0.875))
    ablation = taca_spec(first_n_steps=2)
    assert ablation.window == StepWindow(end=2)
    assert ablation.active(0, make_ctx(sigma=0.875)) and not ablation.active(0, make_ctx(sigma=0.75))


def test_taca_logit_scale_sign_monotonicity():
    s = torch.randn(2, 8, 24, 24, generator=torch.Generator().manual_seed(13))
    layout = SeqLayout("img_txt", 16, 8)
    qm, km = index_mask(24, layout.q_img), index_mask(24, layout.k_txt)
    region = qm[:, None] & km[None, :]
    full = slice(0, 24)
    outs = [LogitEdit(Region(query=qm, key=km), "scale", g)(s, full, full) for g in (1.0, 1.1, 1.2, 1.5, 2.0)]
    pos, neg = (s > 0) & region, (s < 0) & region
    for a, b in zip(outs, outs[1:]):
        assert bool((b[pos] >= a[pos]).all()) and bool((b[neg] <= a[neg]).all())
        assert torch.equal(b[:, :, ~region], s[:, :, ~region])
    assert torch.equal(torch.sign(outs[-1]), torch.sign(s))  # scaling never flips a logit's sign


def test_logit_bias_monotonically_raises_text_mass(layouts, make_qkv):
    """Sign-independent log-bias ``s' = s + delta`` on img->txt raises every image row's text share."""
    layout = layouts["img_txt"]
    q, k, _ = make_qkv(layout)
    qm, km = index_mask(24, layout.q_img), index_mask(24, layout.k_txt)
    masses = []
    for delta in (0.0, 0.5, 1.0, 2.0):
        p = torch.softmax(LogitEdit(Region(query=qm, key=km), "add", delta)(
            (q @ expand_kv(k, 4).transpose(-1, -2)) * 0.25, slice(0, 24), slice(0, 24)), -1)
        masses.append(p[:, :, layout.q_img, layout.k_txt].sum(-1))
    for a, b in zip(masses, masses[1:]):
        assert bool((b > a).all())


def test_taca_session_hook_only_in_window(capabilities, layouts, make_qkv, make_ctx, native_sdpa):
    layout = layouts["img_txt"]
    q, k, v = make_qkv(layout)
    sess = XAISession(capabilities["moss"], interventions=[taca_spec(gamma0=1.2)])
    native = native_sdpa()
    lo = sess.attention(q, k, v, block=0, ctx=make_ctx(sigma=0.5), layout=layout, native=native)
    assert native.calls == 1 and maxdiff(lo, native(q, k, v)) == 0
    hi = sess.attention(q, k, v, block=0, ctx=make_ctx(sigma=1.0), layout=layout, native=native)
    qm, km = index_mask(24, layout.q_img), index_mask(24, layout.k_txt)
    ref = chunked_attention(q, k, v, logit_edits=[LogitEdit(Region(query=qm, key=km), "scale", 1.2)])
    assert maxdiff(hi, ref) < 1e-6 and maxdiff(hi, lo) > 1e-4


@pytest.mark.parametrize("order", ["img_txt", "txt_img"])
@pytest.mark.parametrize("keys", [None, (TokenSpan(1, 4),)])
def test_taca_listing2_restores_text_queries(layouts, make_qkv, order, keys):
    layout = layouts[order]
    q, k, v = make_qkv(layout)
    gamma = 1.3
    out = taca_listing2_attention(q, k, v, layout, gamma, keys=keys)
    kmask = (index_mask(layout.n_key, layout.k_txt) if keys is None
             else union_mask(layout.n_key, [layout.txt_keys(sp) for sp in keys]))
    qmask = index_mask(layout.n_query, layout.q_img)
    ref = chunked_attention(q, k, v, logit_edits=[LogitEdit(Region(query=qmask, key=kmask), "scale", gamma)])
    base = chunked_attention(q, k, v)
    assert maxdiff(out, ref) < 1e-5  # image rows == img x keys logit scaling
    assert maxdiff(out[:, :, layout.q_txt], base[:, :, layout.q_txt]) < 1e-5  # text rows restored
    assert maxdiff(out[:, :, layout.q_img], base[:, :, layout.q_img]) > 1e-4
    assert maxdiff(taca_listing2_attention(q, k, v, layout, 1.0), base) < 1e-5


# =========================================================================== E3-4 DiT-ST / SAP


def test_attention_change_and_tracker():
    a = torch.rand(2, 8, 16, 8, generator=torch.Generator().manual_seed(14))
    assert attention_change(a, a).abs().max().item() == 0
    assert maxdiff(attention_change(a, 2 * a), torch.ones(2, 8)) < 1e-6
    tr = AttentionChangeTracker()
    for blk in (0, 1):
        tr.add(blk, 0, a)
        tr.add(blk, 1, a)
        tr.add(blk, 2, 2 * a)
    series = tr.series()
    assert list(series) == [1, 2]
    assert series[1] == pytest.approx(0.0, abs=1e-9) and series[2] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        tr.add(0, 4, a)


def test_normalized_boundary_depends_on_dsigma():
    raw = {1: 0.01, 2: 0.01, 3: 0.01}
    sigmas = [1.0, 0.99, 0.5, 0.49]
    norm = normalize_by_dsigma(raw, sigmas)
    assert norm[1] == pytest.approx(1.0) and norm[2] == pytest.approx(0.01 / 0.49) and norm[3] == pytest.approx(1.0)
    assert attribute_boundary(norm, tau=0.05) == (2, "tau")
    assert attribute_boundary(raw, tau=0.05) == (1, "tau")
    assert normalize_by_dsigma({0: 1.0, 1: 1.0}, [1.0, 1.0]) == {}  # i=0 and zero steps skipped


def test_attribute_boundary_fallbacks():
    assert attribute_boundary({1: 0.5, 2: 0.3, 3: 5e-5, 4: 1e-5}) == (3, "tau")
    assert attribute_boundary({1: 1.0, 2: 2.0, 3: 0.5, 4: 0.15}) == (4, "relative_drop")
    assert attribute_boundary({1: 1.0, 2: 1.0}) == (None, "none")


def test_moving_average_and_curvature():
    assert moving_average({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, w=3) == pytest.approx({0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0})
    assert discrete_curvature([0.0, 1.0, 2.0, 3.0]) == [0.0, 0.0, 0.0, 0.0]
    kappa = discrete_curvature([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    assert max(range(6), key=kappa.__getitem__) == 2
    assert snr([0.5, 1.0, 0.0], eps_sigma=1e-4)[:2] == [1.0, 0.0]


@pytest.mark.parametrize("grid", ["index", "uniform"])
@pytest.mark.parametrize("x", [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]])
def test_relation_boundary_kink(grid, x):
    y = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 2.0, 5: 3.0}
    assert relation_boundary(y, x, s_attr=6, grid=grid) == 2
    assert relation_boundary(y, x, s_attr=2, grid=grid) is None  # < 3 prefix points


def test_sap_stage_prompt_schedule(make_ctx):
    b = [CondBundle(text_emb=torch.zeros(3, 4), token_len=3), CondBundle(text_emb=torch.ones(5, 4), token_len=5),
         CondBundle(text_emb=torch.full((4, 4), 2.0), token_len=4)]
    assert StagePromptSchedule(b[:2], switch_steps=[3]).sequence(SIGMAS) == [0, 0, 0, 1, 1, 1, 1, 1]
    assert StagePromptSchedule(b[:2], switch_sigmas=[0.6]).sequence(SIGMAS) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert StagePromptSchedule(b, switch_steps=[2, 5]).sequence(SIGMAS) == [0, 0, 1, 1, 1, 2, 2, 2]
    sched = StagePromptSchedule(b[:2], switch_steps=[3])
    assert sched.bundle_at(make_ctx(sigma=0.5)) is b[1] and sched.bundle_at(make_ctx(sigma=0.75)) is b[0]
    with pytest.raises(ValueError):
        StagePromptSchedule(b[:2], switch_steps=[8]).sequence(SIGMAS)


@pytest.mark.parametrize("kw", [
    dict(bundles=[]),
    dict(switch_steps=[3], switch_sigmas=[0.5]),
    dict(switch_steps=[3, 4]),
    dict(switch_steps=[0]),
    dict(switch_sigmas=[1.2]),
])
def test_sap_schedule_validation(kw):
    kw = dict(kw)
    bundles = kw.pop("bundles", [CondBundle(torch.zeros(1), 1), CondBundle(torch.zeros(1), 1)])
    with pytest.raises(ValueError):
        StagePromptSchedule(bundles, **kw)
    with pytest.raises(ValueError):
        StagePromptSchedule([CondBundle(torch.zeros(1), 1)] * 3, switch_sigmas=[0.3, 0.6])


# =========================================================================== offline BF16 cond cache


def test_cond_cache_bf16_roundtrip(tmp_path):
    calls = []

    def encode(prompt):
        calls.append(prompt)
        return {"emb": torch.randn(3, 4, generator=torch.Generator().manual_seed(0)),
                "ids": torch.tensor([1, 2, 3]), "len": 3}

    cache = CondCache(tmp_path / "te", "dummy-encoder")
    a = cache.get_or_encode("hello", encode)
    b = cache.get_or_encode("hello", encode)
    assert calls == ["hello"]
    assert a["emb"].dtype == torch.bfloat16 and a["ids"].dtype == torch.int64 and b["len"] == 3
    assert torch.equal(a["emb"], b["emb"])
    assert cache.key("hello") != cache.key("hello", neg=True) != cache.key("world")
    assert cache.manifest_hash(["a", "b"]) == cache.manifest_hash(["a", "b"]) != cache.manifest_hash(["b", "a"])
    assert not list((tmp_path / "te").glob("*.tmp"))
