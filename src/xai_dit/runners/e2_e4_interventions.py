"""E2-2 / E3-3 / E3-1 / E4-3 Before/After intervention runner (Colab L4; Anima, MOSS-Image, Krea 2).

Experiments (each wrapped in its own ``try/except`` per model; one failure never aborts the rest):

* **E2-2** ``Cont^(l)`` layer ranking for the first localizable manifest target (ONE hooked pass,
  ``adapter.output_projection(i)`` called as a module, never ``wo.weight``; Krea 2 ranks by
  ``gated_mvo``), then Top-20% vs Bottom-20% target K/V substitution with the neutral prompt
  (``kv_patch``; needs strict token alignment) or ``logit_knockout`` fallback when the neutral
  prompt is not length/ID-aligned.
* **E3-3** TACA ``logit_scale`` (``gamma0`` on ``sigma >= sigma_thresh``, cond only, img -> ``S`` keys).
* **E3-1** TextGen K/V patching of a colour attribute (length-matched src/tgt pair checked with
  ``assert_strict_token_alignment``) on Early / Mid / Late block thirds.
* **E4-3** Krea 2 output-gate ablation (blocks ``0..5``: ``gate_override`` ``one`` and ``mean``)
  with downstream per-block M-Act (``max |attn_out|`` and ``max |residual_out|``, cond branch).

Policies: target spans are resolved with the lightweight tokenizer (zero TE weights; the TE is
only loaded by ``ensure_cond_cache`` on a conditioning-cache miss); all per-block accumulators
stay on the GPU (no ``.item()`` / host syncs inside attention chunk loops, one transfer per pass);
every stage and sampling step is logged with ``flush=True``.

Outputs (``experiments/design/output_spec.md`` layout)::

    <out_root>/e2_2_<model>_<precision>_<run_id>/        visuals/e2_2_cont_ranking.png, raw/cont.npz
    <out_root>/e3_3_<model>_<precision>_<run_id>/        visuals/e3_3_taca_compare.png
    <out_root>/e3_1_<model>_<precision>_<run_id>/        visuals/e3_1_kv_patch_grid.png
    <out_root>/e4_3_ablation_krea2_fp8_<run_id>/         visuals/e4_3_krea2_gate_ablation.png, raw/mact.npz
    (each with metrics.json + config.json)
    <out_root>/cross_model/summary_e2_e4_interventions_<run_id>.json
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from .cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    SCHEMA_VERSION,
    bootstrap,
    empty_latent,
    ensure_cond_cache,
    install_runtime_wrappers,
    load_manifest,
    log,
    pick_prompt,
    replace_occurrence,
    resolve_weight,
    save_png,
    to_conditioning,
    token_map_json,
    unload_all,
    weight_precision,
    with_shift,
    write_json,
)
from .e1_e4_profile import _load_lightweight_tokenizer, _match_target_spans_in_S

EXPERIMENTS = ("e2_2", "e3_3", "e3_1", "e4_3")
MODEL_ORDER = ("anima", "moss_image", "krea2")
PROMPT_MAP = {"anima": "P-A1", "moss_image": "P-M1", "krea2": "P-K1"}

E3_1_TEMPLATE = "A portrait of a young woman wearing a {} scarf in a sunlit cafe"
E3_1_COLOR_PAIRS = (("red", "blue"), ("red", "green"), ("black", "white"), ("yellow", "purple"))
E3_1_MAX_CANDIDATES = 3  # pre-qualified pairs encoded in the single TE pass

# Validated reference categorical palette (dataviz palette.md, light mode), fixed slot order.
C_SLOT1, C_SLOT2, C_SLOT3 = "#2a78d6", "#eb6834", "#1baf7a"
C_NEUTRAL, C_INK2 = "#c3c2b7", "#52514e"


# =========================================================================== small helpers


def _exp_dir(out_root: Path, exp: str, model: str, run_id: str) -> Path:
    if exp == "e4_3":
        return out_root / f"e4_3_ablation_{model}_{MODEL_DEFAULTS[model].precision}_{run_id}"
    return out_root / f"{exp}_{model}_{MODEL_DEFAULTS[model].precision}_{run_id}"


def _prep_dir(path: Path) -> Path:
    (path / "raw").mkdir(parents=True, exist_ok=True)
    (path / "visuals").mkdir(parents=True, exist_ok=True)
    return path


def _np_img(image: Any) -> Any:
    return image.float().clamp(0, 1).cpu().numpy()


def _img_delta(a: Any, b: Any) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max())}


def _thirds(num_blocks: int) -> dict[str, list[int]]:
    b1, b2 = round(num_blocks / 3), round(2 * num_blocks / 3)
    return {"early": list(range(0, b1)), "mid": list(range(b1, b2)), "late": list(range(b2, num_blocks))}


def _precheck_color_pairs(tok: Any) -> list[tuple[str, str]]:
    """Tokenizer-only pre-check: same length and one contiguous differing region (zero TE weights)."""
    ok: list[tuple[str, str]] = []
    for a, b in E3_1_COLOR_PAIRS:
        ia = [int(t) for t in tok.encode(E3_1_TEMPLATE.format(a), add_special_tokens=False)]
        ib = [int(t) for t in tok.encode(E3_1_TEMPLATE.format(b), add_special_tokens=False)]
        if len(ia) != len(ib):
            log(f"E3-1 pair ({a}, {b}) rejected: token length {len(ia)} != {len(ib)}")
            continue
        diff = [i for i, (x, y) in enumerate(zip(ia, ib)) if x != y]
        if not diff or diff[-1] - diff[0] + 1 != len(diff):
            log(f"E3-1 pair ({a}, {b}) rejected: differing positions {diff} not one contiguous run")
            continue
        ok.append((a, b))
        if len(ok) >= E3_1_MAX_CANDIDATES:
            break
    if not ok:
        log("WARNING E3-1: no colour pair passed the tokenizer pre-check; trying the first pair anyway")
        ok.append(E3_1_COLOR_PAIRS[0])
    return ok


# =========================================================================== GPU-only observers / sessions


def _make_cont_tracker(torch: Any, num_blocks: int, device: Any, span: Any,
                       wo_for_block: Callable[[int], Any]) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    """Streaming per-block ``Cont^(l)`` for one text span (cond branch), accumulated on the GPU.

    Per attention call only ``A[img, span] @ V[span]`` is accumulated (no full probabilities);
    ``wo`` is applied once in ``finish`` via :func:`contribution_from_context` (module call).
    Chunk callbacks use pure-Python slice arithmetic only (no host syncs).
    """
    from ..core.layer_selection import contribution_from_context
    from ..core.session import ProbObserver, grouped_pv, slice_overlap

    acc = {"raw_mvo": torch.zeros(num_blocks, dtype=torch.float64, device=device),
           "gated_mvo": torch.zeros(num_blocks, dtype=torch.float64, device=device),
           "count": [0] * num_blocks, "gated_count": [0] * num_blocks}

    class _ContObs(ProbObserver):
        def __init__(self, call: Any, row: int) -> None:
            lay = call.layout
            self.call, self.row, self.q_img = call, row, lay.q_img
            self.k_span = lay.txt_keys(span)
            self.v = call.v[row:row + 1, :, self.k_span].float()  # [1, Hkv, |I|, d] (pre GQA repeat)
            self.buf = torch.zeros(1, call.q.shape[1], lay.n_img, call.v.shape[-1],
                                   dtype=torch.float32, device=call.device)

        def __call__(self, probs: Any, q_sl: slice, k_sl: slice) -> None:
            qo = slice_overlap(q_sl, self.q_img)
            ko = slice_overlap(k_sl, self.k_span)
            if qo is None or ko is None:
                return
            p = probs[self.row:self.row + 1, :, qo[0], ko[0]].float()
            self.buf[:, :, qo[1]] += grouped_pv(p, self.v[:, :, ko[1]])

        def finish(self) -> None:
            lay, ex = self.call.layout, self.call.extras

            def img_rows(t: Any) -> Any:
                if t is None:
                    return None
                t = t[self.row:self.row + 1]
                return t[:, lay.q_img] if t.ndim >= 2 and t.shape[1] == lay.n_query else t

            ctx_o = self.buf.transpose(1, 2).flatten(2)  # [1, n_img, Hq*d]
            res = contribution_from_context(ctx_o, wo_for_block(self.call.block),
                                            gate_logits=img_rows(ex.get("gate_logits")),
                                            g1=img_rows(ex.get("g1")), wo_dtype=self.call.v.dtype)
            b = self.call.block
            acc["raw_mvo"][b] += res["raw_mvo"][0].double()
            acc["count"][b] += 1
            if "gated_mvo" in res:
                acc["gated_mvo"][b] += res["gated_mvo"][0].double()
                acc["gated_count"][b] += 1
            self.buf = self.v = None

    def factory(call: Any) -> Optional[Any]:
        rows = call.ctx.rows("cond")
        return _ContObs(call, rows[0]) if rows else None

    return factory, acc


def _make_mact_session(torch: Any, capability: Any, device: Any, interventions: Sequence[Any],
                       query_chunk: int) -> tuple[Any, dict[str, Any]]:
    """Session streaming per-block M-Act (``max |attn_out|``, ``max |residual_out|``) and mean
    effective ``gate_value`` of the cond branch into GPU accumulators (``_store`` override)."""
    from ..adapters.base import CaptureSpec
    from ..core.session import XAISession

    L = capability.num_blocks
    acc = {"attn_max": torch.zeros(L, dtype=torch.float32, device=device),
           "attn_stepmax_sum": torch.zeros(L, dtype=torch.float64, device=device),
           "res_max": torch.zeros(L, dtype=torch.float32, device=device),
           "res_stepmax_sum": torch.zeros(L, dtype=torch.float64, device=device),
           "gate_mean_sum": torch.zeros(L, dtype=torch.float64, device=device),
           "attn_n": [0] * L, "res_n": [0] * L, "gate_n": [0] * L}
    fields = {"attn_out", "residual_out"} | ({"gate_value"} if capability.has_output_gate else set())

    class _MActSession(XAISession):
        def _store(self, name: str, tensor: Any, block: int, ctx: Any) -> None:
            t = tensor.detach()
            if name in ("attn_out", "residual_out"):
                key = "attn" if name == "attn_out" else "res"
                m = t.float().abs().amax()
                acc[f"{key}_max"][block] = torch.maximum(acc[f"{key}_max"][block], m)
                acc[f"{key}_stepmax_sum"][block] += m.double()
                acc[f"{key}_n"][block] += 1
            elif name == "gate_value":
                acc["gate_mean_sum"][block] += t.float().mean().double()
                acc["gate_n"][block] += 1

    sess = _MActSession(capability, capture=CaptureSpec(fields=frozenset(fields), branches=("cond",), to_cpu=False),
                        interventions=list(interventions), query_chunk=query_chunk)
    return sess, acc


def _mact_to_numpy(np: Any, acc: Mapping[str, Any]) -> dict[str, Any]:
    out = {"attn_max": acc["attn_max"].cpu().numpy().astype(np.float64),
           "res_max": acc["res_max"].cpu().numpy().astype(np.float64),
           "attn_stepmax_mean": acc["attn_stepmax_sum"].cpu().numpy() / np.maximum(np.array(acc["attn_n"]), 1),
           "res_stepmax_mean": acc["res_stepmax_sum"].cpu().numpy() / np.maximum(np.array(acc["res_n"]), 1)}
    if any(acc["gate_n"]):
        out["gate_mean"] = acc["gate_mean_sum"].cpu().numpy() / np.maximum(np.array(acc["gate_n"]), 1)
    return out


# =========================================================================== per-model runtime


class _ModelRun:
    """Loaded DiT + VAE + conditioning for one model; :meth:`generate` logs every sampling step."""

    def __init__(self, c: SimpleNamespace, args: argparse.Namespace, model: str, experiments: Sequence[str]) -> None:
        import torch
        from ..adapters import get_adapter

        self.c, self.args, self.name, self.torch = c, args, model, torch
        self.d = d = MODEL_DEFAULTS[model]
        self.prompt_id = PROMPT_MAP[model]
        self.manifest, self.manifest_sha = load_manifest(REPO_ROOT / f"experiments/design/manifest_{model}.json")
        self.entry = pick_prompt(self.manifest, self.prompt_id)
        self.prompt = self.entry["prompt"]
        self.negative = self.entry.get("negative_prompt", self.manifest.get("default_negative_prompt", d.negative))
        self.seed = int(args.seed)
        self.adapter = get_adapter(model)
        self.cap = self.adapter.capability
        self.base_image: Any = None
        self.t_model = time.monotonic()

        # ---- 1. tokenizer-only planning (zero TE weights)
        log(f"[{model}] stage 1/4: lightweight tokenizer span planning")
        tok = _load_lightweight_tokenizer(model)
        self.target = next((t for t in self.entry.get("targets", []) if t.get("localizable")), None)
        self.neutral_prompt: Optional[str] = None
        if self.target is not None and self.target.get("neutral_sub"):
            self.neutral_prompt = replace_occurrence(self.prompt, self.target["text"], self.target["neutral_sub"],
                                                     int(self.target.get("occurrence_idx", 0)))
        self.color_candidates = _precheck_color_pairs(tok) if "e3_1" in experiments else []
        texts = [self.prompt, self.negative]
        if self.neutral_prompt and "e2_2" in experiments:
            texts.append(self.neutral_prompt)
        for a, b in self.color_candidates:
            texts += [E3_1_TEMPLATE.format(a), E3_1_TEMPLATE.format(b)]
        self.texts = list(dict.fromkeys(texts))

        # ---- 2. conditioning (TE loaded once, only on cache misses, then unloaded)
        log(f"[{model}] stage 2/4: conditioning cache for {len(self.texts)} texts")
        cc_args = SimpleNamespace(te=args.te, models_dir=args.models_dir, cache_root=str(args.cache_root))
        with torch.inference_mode():
            self.cache, self.conds = ensure_cond_cache(c, cc_args, d, self.adapter, self.texts, self.manifest_sha)
        self.tm = self.adapter.build_token_map(self.conds[self.prompt]["token_ids"], branch="cond")

        # ---- 3. span resolution on the real DiT token IDs (tokenizer only)
        log(f"[{model}] stage 3/4: target spans in S={self.tm.spans['S']}")
        self.target_span: Optional[tuple[int, int]] = None
        self.target_method: Optional[str] = None
        if self.target is not None:
            spans, method = _match_target_spans_in_S(tok, self.tm.token_ids, self.tm.spans["S"], [self.target])
            self.target_span = spans[self.target["text"]]
            self.target_method = method[self.target["text"]]
            log(f"[{model}] E2-2 target {self.target['text']!r} -> {self.target_span} ({self.target_method})")
        self.e3_1_pair = self._select_color_pair(tok) if "e3_1" in experiments else None
        del tok

        # ---- 4. DiT + VAE (quantized weights load natively)
        dit_path = resolve_weight("dit", d.dit, args.models_dir)
        vae_path = resolve_weight("vae", d.vae, args.models_dir)
        log(f"[{model}] stage 4/4: loading DiT {dit_path.name} ({d.precision}) + VAE {vae_path.name}")
        self.dit_path, self.vae_path = dit_path, vae_path
        patcher = c.sd.load_diffusion_model(str(dit_path))
        self.patcher = with_shift(c, patcher, d.shift)
        install_runtime_wrappers(c, self.patcher, d.batching)
        self.adapter.bind(self.patcher)
        self.vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))
        self.sigmas = c.samplers.calculate_sigmas(self.patcher.get_model_object("model_sampling"),
                                                  d.scheduler, d.steps).cpu()
        self.sigma_table = [float(s) for s in self.sigmas]
        self.latent = empty_latent(c, self.patcher)
        self.device = c.mm.get_torch_device()
        log(f"[{model}] ready in {time.monotonic() - self.t_model:.1f}s "
            f"(steps={d.steps}, blocks={self.cap.num_blocks}, latent={list(self.latent.shape)})")

    # ------------------------------------------------------------------ planning helpers
    def _select_color_pair(self, tok: Any) -> Optional[dict[str, Any]]:
        from ..adapters.base import TokenAlignmentError, TokenSpan

        for a, b in self.color_candidates:
            src, tgt = E3_1_TEMPLATE.format(a), E3_1_TEMPLATE.format(b)
            tm_s = self.adapter.build_token_map(self.conds[src]["token_ids"], branch="cond")
            tm_t = self.adapter.build_token_map(self.conds[tgt]["token_ids"], branch="cond")
            sa, _ = _match_target_spans_in_S(tok, tm_s.token_ids, tm_s.spans["S"], [{"text": a}])
            sb, _ = _match_target_spans_in_S(tok, tm_t.token_ids, tm_t.spans["S"], [{"text": b}])
            (a0, a1), (b0, b1) = sa[a], sb[b]
            editable = TokenSpan(min(a0, b0), max(a1, b1))
            try:
                tm_s.assert_strict_token_alignment(tm_t, editable=[editable])
            except TokenAlignmentError as e:
                log(f"[{self.name}] E3-1 pair ({a}, {b}) failed strict alignment: {e}")
                continue
            log(f"[{self.name}] E3-1 pair ({a} -> {b}) aligned: len={tm_s.text_len}, editable={editable}")
            return {"src_color": a, "tgt_color": b, "src": src, "tgt": tgt, "editable": editable,
                    "tm_src": tm_s, "tm_tgt": tm_t}
        return None

    def kv_cache_to_cpu(self, n_txt: int, n_blocks: int) -> tuple[bool, float]:
        """Cond-only text K/V cache size estimate (fp32 upper bound) vs the GPU budget."""
        gb = n_txt * self.cap.num_kv_heads * self.cap.head_dim * 2 * 4 * n_blocks * self.d.steps / 1e9
        mode = self.args.kv_cache_device
        return (mode == "cpu" or (mode == "auto" and gb > self.args.kv_gpu_budget_gb)), gb

    # ------------------------------------------------------------------ generation
    def generate(self, tag: str, *, session: Any = None, positive_text: Optional[str] = None) -> Any:
        c, d = self.c, self.d
        positive = to_conditioning(self.conds[positive_text or self.prompt])
        negative = to_conditioning(self.conds[self.negative])
        t0 = last = time.monotonic()
        log(f"[{self.name}] >>> generate '{tag}' ({'hooked' if session is not None else 'plain'})")

        def cb(step: int, x0: Any, x: Any, total: int) -> None:  # host clock only, no CUDA sync
            nonlocal last
            now = time.monotonic()
            log(f"[{self.name}]     {tag} step {step + 1}/{total} (σ={self.sigma_table[step]:.3f}) +{now - last:.1f}s")
            last = now

        noise = c.sample.prepare_noise(self.latent, self.seed)
        sampler = c.samplers.sampler_object(d.sampler)
        run = lambda: c.sample.sample_custom(self.patcher, noise, d.cfg, sampler, self.sigmas, positive, negative,  # noqa: E731
                                             self.latent, callback=cb, disable_pbar=True, seed=self.seed)
        if session is None:
            samples = run()
        else:
            with self.adapter.attached(session):
                samples = run()
        img = self.vae.decode(samples)
        image = img.reshape(-1, *img.shape[-3:])[0].detach().float().cpu()
        log(f"[{self.name}] <<< '{tag}' done in {time.monotonic() - t0:.1f}s")
        return image

    def base(self) -> Any:
        if self.base_image is None:
            self.base_image = self.generate("base")
        return self.base_image

    def session(self, interventions: Sequence[Any] = ()) -> Any:
        from ..core.session import XAISession
        return XAISession(self.cap, interventions=list(interventions), query_chunk=self.args.query_chunk)

    # ------------------------------------------------------------------ shared config
    def config(self, exp_id: str, run_id: str, **extra: Any) -> dict[str, Any]:
        d = self.d
        return {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "exp_id": exp_id,
            "model": {"name": self.name, "precision": d.precision, "dit": self.dit_path.name, "te": Path(self.args.te or d.te).name,
                      "te_precision": weight_precision(self.args.te or d.te), "vae": self.vae_path.name,
                      "te_cache_sha256": self.cache.manifest_hash(self.texts)},
            "runtime": {"sampler": d.sampler, "scheduler": d.scheduler, "steps": d.steps, "cfg": d.cfg,
                        "shift": d.shift, "cfg_batching_policy": d.batching, "seed": self.seed,
                        "query_chunk": self.args.query_chunk, "latent_shape": list(self.latent.shape),
                        "sigma_table": self.sigma_table},
            "prompt": {"prompt_id": self.prompt_id, "prompt": self.prompt, "negative_prompt": self.negative},
            "token_map": token_map_json(self.tm),
            **extra,
        }

    def close(self) -> None:
        for attr in ("patcher", "vae", "base_image"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        unload_all(self.c)


# =========================================================================== experiments


def run_e2_2(mr: _ModelRun, out_root: Path, run_id: str) -> dict[str, Any]:
    import numpy as np
    torch = mr.torch
    from ..adapters.base import InterventionSpec, TokenAlignmentError, TokenSpan
    from ..core.layer_selection import top_bottom_random
    from ..core.patching import KVCache

    if mr.target is None or mr.target_span is None:
        return {"status": "n/a", "reason": "no localizable target in the manifest entry"}
    span = TokenSpan(*mr.target_span)
    L = mr.cap.num_blocks
    t0 = time.monotonic()

    # 1. Cont^(l) in ONE hooked pass (cond branch, every step, every block)
    log(f"[{mr.name}] E2-2 (1/4) Cont^(l) pass for {mr.target['text']!r} span={span}")
    factory, acc = _make_cont_tracker(torch, L, mr.device, span, mr.adapter.output_projection)
    sess = mr.session()
    sess.add_observer(factory)
    cont_img = mr.generate("e2_2_cont_pass", session=sess)
    raw = acc["raw_mvo"].cpu().numpy() / np.maximum(np.array(acc["count"]), 1)  # single GPU->CPU transfer
    gated = (acc["gated_mvo"].cpu().numpy() / np.maximum(np.array(acc["gated_count"]), 1)
             if any(acc["gated_count"]) else None)
    rank_metric = "gated_mvo" if gated is not None else "raw_mvo"
    scores = gated if gated is not None else raw
    sets = top_bottom_random(torch.from_numpy(scores), mr.args.top_frac, generator=torch.Generator().manual_seed(mr.seed))
    log(f"[{mr.name}] E2-2 ranking by {rank_metric}: top={sets['top']} bottom={sets['bottom']}")

    # 2. substitution mode: neutral K/V (strictly aligned) else logit knockout
    mode, align_error = "logit_knockout", None
    if mr.neutral_prompt is not None:
        tm_n = mr.adapter.build_token_map(mr.conds[mr.neutral_prompt]["token_ids"], branch="cond")
        try:
            mr.tm.assert_strict_token_alignment(tm_n, editable=[span])
            mode = "kv_patch_neutral"
        except TokenAlignmentError as e:
            align_error = str(e)
            log(f"[{mr.name}] E2-2 neutral prompt not aligned ({e}); falling back to logit_knockout")

    base = mr.base()
    images: dict[str, Any] = {}
    neutral_img, kv_cpu, kv_gb = None, None, None
    if mode == "kv_patch_neutral":
        blocks = sorted(set(sets["top"]) | set(sets["bottom"]))
        kv_cpu, kv_gb = mr.kv_cache_to_cpu(mr.tm.text_len, len(blocks))
        cache = KVCache("txt", to_cpu=kv_cpu)
        log(f"[{mr.name}] E2-2 (2/4) neutral K/V store pass on blocks {blocks} "
            f"(~{kv_gb:.2f} GB, {'cpu' if kv_cpu else 'gpu'})")
        store = InterventionSpec("kv_patch", blocks=blocks, cfg_scope="cond_only",
                                 params={"source": cache, "mode": "store"})
        neutral_img = mr.generate("e2_2_neutral_store", session=mr.session([store]), positive_text=mr.neutral_prompt)
        for n, arm in enumerate(("top", "bottom"), 3):
            log(f"[{mr.name}] E2-2 ({n}/4) neutral K/V substitution on {arm}-20% blocks {sets[arm]}")
            spec = InterventionSpec("kv_patch", blocks=sets[arm], cfg_scope="cond_only",
                                    params={"source": cache, "keys": (span,)})
            images[arm] = mr.generate(f"e2_2_{arm}", session=mr.session([spec]))
        del cache
    else:
        for n, arm in enumerate(("top", "bottom"), 3):
            log(f"[{mr.name}] E2-2 ({n}/4) logit_knockout on {arm}-20% blocks {sets[arm]}")
            spec = InterventionSpec("logit_knockout", blocks=sets[arm], cfg_scope="cond_only",
                                    params={"keys": (span,), "query": "img"})
            images[arm] = mr.generate(f"e2_2_{arm}", session=mr.session([spec]))

    out = _prep_dir(_exp_dir(out_root, "e2_2", mr.name, run_id))
    for tag, img in (("base", base), ("cont_pass", cont_img), ("top20", images["top"]),
                     ("bottom20", images["bottom"]), ("neutral", neutral_img)):
        if img is not None:
            save_png(img, out / "visuals" / f"{tag}.png")
    _render_e2_2(mr, raw, gated, rank_metric, sets, base, images, mode, out / "visuals" / "e2_2_cont_ranking.png")
    np.savez_compressed(out / "raw" / "cont.npz", raw_mvo=raw, **({"gated_mvo": gated} if gated is not None else {}),
                        top=np.array(sets["top"]), bottom=np.array(sets["bottom"]), random=np.array(sets["random"]),
                        count=np.array(acc["count"]))
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "model": mr.name, "prompt_id": mr.prompt_id,
        "status": "ok", "elapsed_sec": time.monotonic() - t0,
        "target": mr.target["text"], "target_span": [span.start, span.end], "target_span_method": mr.target_method,
        "rank_metric": rank_metric, "cont_raw_mvo": raw.tolist(), "cont_gated_mvo": None if gated is None else gated.tolist(),
        "blocks": sets, "top_frac": mr.args.top_frac, "substitution_mode": mode, "alignment_error": align_error,
        "pixel_delta_vs_base": {"top20": _img_delta(images["top"], base), "bottom20": _img_delta(images["bottom"], base),
                                "cont_pass_path_check": _img_delta(cont_img, base)},
    }
    metrics["top_minus_bottom_mean_abs"] = (metrics["pixel_delta_vs_base"]["top20"]["mean_abs"]
                                            - metrics["pixel_delta_vs_base"]["bottom20"]["mean_abs"])
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", mr.config("e2_2", run_id, e2_2={
        "target": mr.target, "neutral_prompt": mr.neutral_prompt, "substitution_mode": mode,
        "cfg_scope": "cond_only", "query": "img", "kv_patch_positions": "target span only (text K/V)",
        "kv_cache_cpu": kv_cpu, "kv_cache_est_gb": kv_gb, "cont_branch": "cond", "cont_steps": "all",
        "wo": "adapter.output_projection(i) module call", "random_set_seed": mr.seed}))
    log(f"[{mr.name}] E2-2 done -> {out.name} ({mode}; Δtop={metrics['pixel_delta_vs_base']['top20']['mean_abs']:.4f}, "
        f"Δbottom={metrics['pixel_delta_vs_base']['bottom20']['mean_abs']:.4f})")
    return {"status": "ok", "out_dir": out.name, "substitution_mode": mode, "rank_metric": rank_metric,
            "blocks": sets, "top_minus_bottom_mean_abs": metrics["top_minus_bottom_mean_abs"]}


def run_e3_3(mr: _ModelRun, out_root: Path, run_id: str) -> dict[str, Any]:
    from ..core.patching import taca_spec

    gamma0, thresh = mr.args.taca_gamma, mr.args.taca_sigma
    active = [i for i in range(mr.d.steps) if mr.sigma_table[i] >= thresh]
    if not active:
        log(f"[{mr.name}] WARNING E3-3: no step has σ >= {thresh}; TACA would be a no-op, skipping")
        return {"status": "n/a", "reason": f"no sampling step with sigma >= {thresh}"}
    t0 = time.monotonic()
    span_S = mr.tm.spans["S"]
    log(f"[{mr.name}] E3-3 TACA γ0={gamma0} on steps {active} (σ>={thresh}), cond_only, keys S={span_S}")
    base = mr.base()
    spec = taca_spec(keys=span_S, gamma0=gamma0, sigma_thresh=thresh, cfg_scope="cond_only")
    taca = mr.generate("e3_3_taca", session=mr.session([spec]))

    out = _prep_dir(_exp_dir(out_root, "e3_3", mr.name, run_id))
    save_png(base, out / "visuals" / "base.png")
    save_png(taca, out / "visuals" / f"taca_{gamma0:g}.png")
    diff = ((taca - base).abs() * 5).clamp(0, 1)
    _render_panels([(base, "Base"), (taca, f"TACA γ0={gamma0:g} (σ≥{thresh:g}, steps {active[0]}..{active[-1]})"),
                    (diff, "|Δ| × 5")],
                   f"{mr.name} ({mr.prompt_id}) — E3-3 TACA early-step text-attention scaling",
                   out / "visuals" / "e3_3_taca_compare.png")
    metrics = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "model": mr.name, "prompt_id": mr.prompt_id,
               "status": "ok", "elapsed_sec": time.monotonic() - t0, "gamma0": gamma0, "sigma_thresh": thresh,
               "active_steps": active, "active_sigmas": [mr.sigma_table[i] for i in active],
               "pixel_delta_vs_base": _img_delta(taca, base)}
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", mr.config("e3_3", run_id, e3_3={
        "hook": "logit_scale", "gamma0": gamma0, "sigma_thresh": thresh, "cfg_scope": "cond_only",
        "query": "img", "keys": "S", "keys_span": [span_S.start, span_S.end], "blocks": "all",
        "attention_path": "manual chunked SDPA on active (block, step); fused elsewhere"}))
    log(f"[{mr.name}] E3-3 done -> {out.name} (Δ mean_abs={metrics['pixel_delta_vs_base']['mean_abs']:.4f})")
    return {"status": "ok", "out_dir": out.name, "active_steps": active,
            "pixel_mean_abs_vs_base": metrics["pixel_delta_vs_base"]["mean_abs"]}


def run_e3_1(mr: _ModelRun, out_root: Path, run_id: str) -> dict[str, Any]:
    from ..adapters.base import InterventionSpec
    from ..core.patching import KVCache

    pair = mr.e3_1_pair
    if pair is None:
        raise RuntimeError(f"no colour pair passed strict token alignment (candidates {mr.color_candidates})")
    t0 = time.monotonic()
    L = mr.cap.num_blocks
    thirds = _thirds(L)
    kv_cpu, kv_gb = mr.kv_cache_to_cpu(pair["tm_src"].text_len, L)
    cache = KVCache("txt", to_cpu=kv_cpu)
    log(f"[{mr.name}] E3-1 ({pair['src_color']} -> {pair['tgt_color']}) source K/V store pass "
        f"(~{kv_gb:.2f} GB, {'cpu' if kv_cpu else 'gpu'})")
    store = InterventionSpec("kv_patch", cfg_scope="cond_only", params={"source": cache, "mode": "store"})
    src_img = mr.generate("e3_1_src_store", session=mr.session([store]), positive_text=pair["src"])
    tgt_img = mr.generate("e3_1_tgt", positive_text=pair["tgt"])
    patched: dict[str, Any] = {}
    for n, (name, blocks) in enumerate(thirds.items(), 1):
        log(f"[{mr.name}] E3-1 patch {n}/3: {name} blocks [{blocks[0]}..{blocks[-1]}]")
        spec = InterventionSpec("kv_patch", blocks=blocks, cfg_scope="cond_only", params={"source": cache})
        patched[name] = mr.generate(f"e3_1_{name}", session=mr.session([spec]), positive_text=pair["tgt"])
    del cache

    out = _prep_dir(_exp_dir(out_root, "e3_1", mr.name, run_id))
    for tag, img in (("src", src_img), ("tgt", tgt_img), *patched.items()):
        save_png(img, out / "visuals" / f"{tag}.png")
    _render_panels([(src_img, f"Src ({pair['src_color']})"), (tgt_img, f"Tgt ({pair['tgt_color']})"),
                    *[(patched[k], f"{k.capitalize()} [{v[0]}..{v[-1]}]") for k, v in thirds.items()]],
                   f"{mr.name} — E3-1 text K/V patch Src→Tgt by block third (cond, all steps)",
                   out / "visuals" / "e3_1_kv_patch_grid.png")
    src_tgt = _img_delta(src_img, tgt_img)["mean_abs"]
    per = {k: {"vs_tgt": _img_delta(v, tgt_img), "vs_src": _img_delta(v, src_img),
               "toward_src_ratio": 1.0 - _img_delta(v, src_img)["mean_abs"] / max(src_tgt, 1e-12)}
           for k, v in patched.items()}
    metrics = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "model": mr.name, "status": "ok",
               "elapsed_sec": time.monotonic() - t0, "pair": [pair["src_color"], pair["tgt_color"]],
               "editable_span": [pair["editable"].start, pair["editable"].end], "thirds": thirds,
               "src_vs_tgt_mean_abs": src_tgt, "patched": per}
    write_json(out / "metrics.json", metrics)
    cfg = mr.config("e3_1", run_id, e3_1={
        "template": E3_1_TEMPLATE, "candidates": mr.color_candidates, "src_prompt": pair["src"], "tgt_prompt": pair["tgt"],
        "alignment": "assert_strict_token_alignment (equal length, equal outer IDs)", "kv_which": "txt",
        "kv_positions": "full text slice", "cfg_scope": "cond_only", "steps": "all",
        "kv_cache_cpu": kv_cpu, "kv_cache_est_gb": kv_gb,
        "token_map_src": token_map_json(pair["tm_src"]), "token_map_tgt": token_map_json(pair["tm_tgt"])})
    cfg["prompt"] = {"src": pair["src"], "tgt": pair["tgt"], "negative_prompt": mr.negative}
    write_json(out / "config.json", cfg)
    log(f"[{mr.name}] E3-1 done -> {out.name} "
        f"""(toward-src: {', '.join(f'{k}={v["toward_src_ratio"]:.2f}' for k, v in per.items())})""")
    return {"status": "ok", "out_dir": out.name, "pair": metrics["pair"],
            "toward_src_ratio": {k: v["toward_src_ratio"] for k, v in per.items()}}


def run_e4_3(mr: _ModelRun, out_root: Path, run_id: str) -> dict[str, Any]:
    import numpy as np
    from ..adapters.base import InterventionSpec

    if not mr.cap.has_output_gate:
        return {"status": "n/a", "reason": f"{mr.name} has no output gate"}
    t0 = time.monotonic()
    blocks = sorted(b for b in mr.args.gate_blocks if 0 <= b < mr.cap.num_blocks)
    arms = {"base": [],
            "gate_one": [InterventionSpec("gate_override", blocks=blocks, params={"mode": "one"})],
            "gate_mean": [InterventionSpec("gate_override", blocks=blocks, params={"mode": "mean"})]}
    images: dict[str, Any] = {}
    mact: dict[str, dict[str, Any]] = {}
    for n, (arm, specs) in enumerate(arms.items(), 1):
        log(f"[{mr.name}] E4-3 arm {n}/3: {arm} (gate blocks {blocks}, M-Act streaming)")
        sess, acc = _make_mact_session(mr.torch, mr.cap, mr.device, specs, mr.args.query_chunk)
        images[arm] = mr.generate(f"e4_3_{arm}", session=sess)
        mact[arm] = _mact_to_numpy(np, acc)  # one GPU->CPU transfer per arm
        del sess, acc

    out = _prep_dir(_exp_dir(out_root, "e4_3", mr.name, run_id))
    for arm, img in images.items():
        save_png(img, out / "visuals" / f"{arm}.png")
    _render_e4_3(mr, images, mact, blocks, out / "visuals" / "e4_3_krea2_gate_ablation.png")
    np.savez_compressed(out / "raw" / "mact.npz", gate_blocks=np.array(blocks),
                        **{f"{arm}_{k}": v for arm, m in mact.items() for k, v in m.items()})
    down = [b for b in range(mr.cap.num_blocks) if b > max(blocks)] if blocks else []

    def summary(m: Mapping[str, Any]) -> dict[str, Any]:
        return {"res_max_peak": float(m["res_max"].max()), "res_max_peak_block": int(m["res_max"].argmax()),
                "attn_max_peak": float(m["attn_max"].max()), "attn_max_peak_block": int(m["attn_max"].argmax()),
                "downstream_res_stepmax_mean": float(m["res_stepmax_mean"][down].mean()) if down else None}

    metrics = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "model": mr.name, "prompt_id": mr.prompt_id,
               "status": "ok", "elapsed_sec": time.monotonic() - t0, "gate_blocks": blocks, "downstream_blocks": down,
               "per_arm": {arm: {"summary": summary(m), **{k: v.tolist() for k, v in m.items()}} for arm, m in mact.items()},
               "pixel_delta_vs_base": {arm: _img_delta(images[arm], images["base"]) for arm in ("gate_one", "gate_mean")}}
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", mr.config("e4_3", run_id, e4_3={
        "hook": "gate_override (post-sigmoid gate_value)", "arms": {"base": "no override", "gate_one": "gate_value <- 1.0",
                                                                    "gate_mean": "gate_value <- per-sample token mean (channel-wise)"},
        "gate_blocks": blocks, "cfg_scope": "both", "mact_branch": "cond",
        "mact_fields": ["attn_out (after gate & wo, before g1)", "residual_out (block output)"],
        "attention_path": "native attention; block slow path on every block for all 3 arms (path-matched)"}))
    log(f"[{mr.name}] E4-3 done -> {out.name}")
    return {"status": "ok", "out_dir": out.name, "gate_blocks": blocks,
            "summary": {arm: summary(m) for arm, m in mact.items()},
            "pixel_mean_abs_vs_base": {k: v["mean_abs"] for k, v in metrics["pixel_delta_vs_base"].items()}}


RUNNERS: dict[str, Callable[[_ModelRun, Path, str], dict[str, Any]]] = {
    "e2_2": run_e2_2, "e3_3": run_e3_3, "e3_1": run_e3_1, "e4_3": run_e4_3}


# =========================================================================== visuals


def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _render_panels(panels: Sequence[tuple[Any, str]], title: str, path: Path) -> None:
    plt = _plt()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), dpi=130)
    for ax, (img, label) in zip(axes if n > 1 else [axes], panels):
        ax.imshow(_np_img(img))
        ax.set_title(label, fontsize=9, color=C_INK2)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)


def _render_e2_2(mr: _ModelRun, raw: Any, gated: Any, rank_metric: str, sets: Mapping[str, list[int]], base: Any,
                 images: Mapping[str, Any], mode: str, path: Path) -> None:
    plt = _plt()
    import numpy as np
    from matplotlib.patches import Patch

    scores = gated if gated is not None else raw
    xs = np.arange(len(scores))
    colors = [C_SLOT1 if b in sets["top"] else C_SLOT2 if b in sets["bottom"] else C_NEUTRAL for b in xs]
    fig = plt.figure(figsize=(14, 9.6), dpi=130)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.7])
    ax = fig.add_subplot(gs[0, :])
    ax.bar(xs, scores, color=colors, width=0.8, edgecolor="white", linewidth=1.0)
    ax.set_xlabel("Block index")
    ax.set_ylabel(f"Cont^(l) ({rank_metric}, mean over steps)")
    ax.set_xticks(xs[::2] if len(xs) > 30 else xs)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=C_SLOT1, label=f"Top-{mr.args.top_frac:.0%}"),
                       Patch(color=C_SLOT2, label=f"Bottom-{mr.args.top_frac:.0%}"),
                       Patch(color=C_NEUTRAL, label="other")], fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"{mr.name} ({mr.prompt_id}) — E2-2 Cont^(l) for {mr.target['text']!r}", fontsize=11)
    sub = "neutral K/V substitution" if mode == "kv_patch_neutral" else "logit knockout (fallback)"
    for j, (img, label) in enumerate([(base, "Base"), (images["top"], f"Top-20% {sets['top']}"),
                                      (images["bottom"], f"Bottom-20% {sets['bottom']}")]):
        a = fig.add_subplot(gs[1, j])
        a.imshow(_np_img(img))
        a.set_title(label if j == 0 else f"{label}\n{sub}", fontsize=9, color=C_INK2)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_e4_3(mr: _ModelRun, images: Mapping[str, Any], mact: Mapping[str, Mapping[str, Any]],
                 blocks: Sequence[int], path: Path) -> None:
    plt = _plt()
    import numpy as np

    arm_style = {"base": (C_SLOT1, "Base"), "gate_one": (C_SLOT2, "gate ← 1.0"), "gate_mean": (C_SLOT3, "gate ← token mean")}
    fig = plt.figure(figsize=(14, 9.8), dpi=130)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0])
    for j, arm in enumerate(arm_style):
        a = fig.add_subplot(gs[0, j])
        a.imshow(_np_img(images[arm]))
        a.set_title(f"{arm_style[arm][1]}" + (f" (blocks {blocks[0]}..{blocks[-1]})" if arm != "base" and blocks else ""),
                    fontsize=9, color=C_INK2)
        a.axis("off")
    for col, (key, ylabel) in enumerate((("res_max", "M-Act: max |residual_out|"), ("attn_max", "M-Act: max |attn_out|"))):
        ax = fig.add_subplot(gs[1, col])
        xs = np.arange(len(mact["base"][key]))
        if blocks:
            ax.axvspan(min(blocks) - 0.5, max(blocks) + 0.5, color=C_NEUTRAL, alpha=0.35, lw=0)
        for arm, (color, label) in arm_style.items():
            ax.plot(xs, mact[arm][key], color=color, lw=2, marker="o", ms=3.5, label=label)
        ax.set_xlabel("Block index (shaded = gate override)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[1, 2])
    if all("gate_mean" in mact[a] for a in arm_style):
        for arm, (color, label) in arm_style.items():
            ax.plot(np.arange(len(mact[arm]["gate_mean"])), mact[arm]["gate_mean"], color=color, lw=2, marker="o",
                    ms=3.5, label=label)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Block index")
        ax.set_ylabel("mean effective gate_value")
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.axis("off")
    fig.suptitle(f"{mr.name} ({mr.prompt_id}) — E4-3 output-gate ablation & downstream M-Act (cond, max over steps)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


# =========================================================================== driver


def _applicable(model: str, exp: str) -> bool:
    return exp != "e4_3" or model == "krea2"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="E2-2 / E3-3 / E3-1 / E4-3 Before/After interventions across models")
    p.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=MODEL_ORDER)
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    p.add_argument("--comfyui-dir", required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=REPO_ROOT / "cache" / "cond")
    p.add_argument("--run-id", default="l4_01")
    p.add_argument("--te", help="override TE file (default per model; only used on conditioning-cache misses)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-chunk", type=int, default=1024)
    p.add_argument("--top-frac", type=float, default=0.2, help="E2-2 Top/Bottom block fraction")
    p.add_argument("--taca-gamma", type=float, default=1.2, help="E3-3 gamma0")
    p.add_argument("--taca-sigma", type=float, default=0.97, help="E3-3 sigma threshold (active while sigma >= it)")
    p.add_argument("--gate-blocks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="E4-3 gate-override blocks")
    p.add_argument("--kv-cache-device", choices=("auto", "gpu", "cpu"), default="auto",
                   help="E2-2/E3-1 K/V cache placement (auto: CPU above --kv-gpu-budget-gb)")
    p.add_argument("--kv-gpu-budget-gb", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true", help="print the plan without importing torch/ComfyUI")
    args = p.parse_args(argv)

    plan = {m: [e for e in args.experiments if _applicable(m, e)] for m in args.models}
    for m, exps in plan.items():
        log(f"plan [{m}] {PROMPT_MAP[m]} ({MODEL_DEFAULTS[m].precision}): "
            f"{exps or 'nothing'} -> {[_exp_dir(args.out_root, e, m, args.run_id).name for e in exps]}")
    if args.dry_run:
        return 0

    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, "models": {}}
    summary_path = args.out_root / "cross_model" / f"summary_e2_e4_interventions_{args.run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    def record(model: str, exp: str, result: dict[str, Any]) -> None:
        summary["models"].setdefault(model, {})[exp] = result
        write_json(summary_path, summary)  # rewritten after every experiment (crash-safe)

    c = bootstrap(SimpleNamespace(comfyui_dir=args.comfyui_dir, enable_dynamic_vram=False, no_highvram=False))
    import torch

    t_all = time.monotonic()
    for n, model in enumerate(args.models, 1):
        for e in args.experiments:
            if not _applicable(model, e):
                record(model, e, {"status": "n/a", "reason": "E4-3 gate ablation is Krea 2 only"})
        exps = plan[model]
        if not exps:
            continue
        log(f"=== model {n}/{len(args.models)}: {model} ({MODEL_DEFAULTS[model].precision}) — {exps} ===")
        mr: Optional[_ModelRun] = None
        try:
            with torch.inference_mode():
                mr = _ModelRun(c, args, model, exps)
                for k, exp in enumerate(exps, 1):
                    log(f"--- [{model}] experiment {k}/{len(exps)}: {exp} ---")
                    t0 = time.monotonic()
                    try:
                        res = RUNNERS[exp](mr, args.out_root, args.run_id)
                    except Exception as ex:  # one failure never aborts the rest
                        log(f"ERROR [{model}] {exp} failed: {type(ex).__name__}: {ex}")
                        print(traceback.format_exc(), flush=True)
                        res = {"status": "error", "error": f"{type(ex).__name__}: {ex}"}
                        c.mm.soft_empty_cache()
                    res["wall_sec"] = time.monotonic() - t0
                    record(model, exp, res)
        except Exception as ex:  # setup failure (conditioning / weights): mark remaining experiments
            log(f"ERROR [{model}] setup failed: {type(ex).__name__}: {ex}")
            print(traceback.format_exc(), flush=True)
            for exp in exps:
                if exp not in summary["models"].get(model, {}):
                    record(model, exp, {"status": "error", "error": f"setup: {type(ex).__name__}: {ex}"})
        finally:
            if mr is not None:
                mr.close()
            else:
                unload_all(c)
            log(f"=== [{model}] finished (+{time.monotonic() - t_all:.1f}s total) ===")

    summary["wall_clock_sec"] = time.monotonic() - t_all
    write_json(summary_path, summary)
    statuses = {m: {e: r.get("status") for e, r in v.items()} for m, v in summary["models"].items()}
    log(f"ALL E2-2/E3-3/E3-1/E4-3 DONE -> {summary_path.name}: {statuses}")
    return 0 if all(s in ("ok", "n/a") for v in statuses.values() for s in v.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
