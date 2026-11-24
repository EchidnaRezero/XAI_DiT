"""Krea 2 Raw adapter (single-stream DiT, ``comfy/ldm/krea2/model.py``; base model only).

Sequence ``[txt T | img N]`` through 28 ``SingleStreamBlock`` (48 Q heads / 12 KV heads x 128,
GQA ``h_kv = h_q // 4``, sigmoid output gate). Hook points:

======================  ======================================================================
``txtfusion.forward``   pre-DiT ``text_in`` swap (``[B, T, 12, 2560]`` 12-layer stack); resets
                        per-forward state
``txtmlp.forward``      pre-DiT ``txt_embed`` swap (DiT text tokens ``[B, T, 6144]``)
``blocks.i.forward``    block slow path (only when the session touches ``(i, step)``):
                        ``kv_patch`` on 12 KV heads after QK-norm / before RoPE, attention via
                        :meth:`XAISession.attention` (K/V passed **pre** ``repeat_interleave``),
                        ``attn_pre_proj`` (= ``ctx``, before gate and ``wo``), ``gate_logits``
                        (raw ``attn.gate(h)``), ``gate_value`` (effective post-sigmoid
                        multiplier, after ``gate_override``), ``attn_out`` (= ``wo(ctx *
                        gate_value)``, before ``g1``), ``residual_out``, ``true_block_bypass``,
                        ``concept_stream``
======================  ======================================================================

``output_projection(i)`` returns the ``attn.wo`` *module* (``wo(x)`` call; never ``wo.weight``,
so FP8 / int8 quantized weights stay valid). Token spans after prefix stripping:
``S=[0, T-5)``, ``R=[T-5, T)`` = ``<|im_end|> \\n <|im_start|> assistant \\n`` (IDs checked).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..core.heatmap import concept_stream_attention
from ..core.session import XAISession
from ._comfy_hooks import (
    ComfyDiTAdapter,
    active_specs,
    gate_override,
    kv_patch,
    maybe_step_context,
    model_namespace,
    original_forward,
    session_touches,
    span_swap,
)
from .base import AdapterCapability, Branch, SeqLayout, StepContext, TokenMap

Tensor = torch.Tensor

KREA2_CAPABILITY = AdapterCapability(
    name="krea2", attention_kind="joint_attn", seq_order="txt_img", num_blocks=28, num_q_heads=48,
    num_kv_heads=12, head_dim=128, text_updated_per_block=True, supports_concept_attention=True,
    has_output_gate=True, text_pad_to=None,
)

# Qwen2.5 / Qwen3-VL tokenizer (``comfy/text_encoders/qwen25_tokenizer``; ``krea2.py`` prefix strip).
QWEN_IM_START = 151644
QWEN_IM_END = 151645
QWEN_USER = 872
QWEN_ASSISTANT = 77091
QWEN_NEWLINE = 198
KREA2_TAIL = (QWEN_IM_END, QWEN_NEWLINE, QWEN_IM_START, QWEN_ASSISTANT, QWEN_NEWLINE)


def strip_template_prefix(full_ids: Sequence[int]) -> list[int]:
    """Replicates ``Krea2TEModel.encode_token_weights``: drop the system turn + ``<|im_start|>user\\n``."""
    ids = [int(t) for t in full_ids]
    template_end, seen = -1, 0
    for j, t in enumerate(ids):
        if t == QWEN_IM_START and seen < 2:
            template_end, seen = j, seen + 1
    if template_end < 0:
        raise ValueError("no <|im_start|> in Krea 2 token IDs")
    if len(ids) > template_end + 3 and ids[template_end + 1] == QWEN_USER and ids[template_end + 2] == QWEN_NEWLINE:
        template_end += 3
    return ids[template_end:]


class Krea2Adapter(ComfyDiTAdapter):
    SWAP_POINTS = frozenset({"text_in", "txt_embed"})

    @property
    def capability(self) -> AdapterCapability:
        return KREA2_CAPABILITY

    # ------------------------------------------------------------------ tokens
    @staticmethod
    def dit_token_ids(tokenized: Mapping[str, Any]) -> list[int]:
        """DiT text-slot IDs from ``clip.tokenize(prompt)`` (prefix stripped as in the TE)."""
        return strip_template_prefix(int(t[0]) for t in tokenized["qwen3vl_4b"][0])

    def build_token_map(self, token_ids: Sequence[int], *, branch: Branch = "cond",
                        targets: Optional[Mapping[str, Any]] = None) -> TokenMap:
        """``token_ids``: DiT slots (prefix-stripped); a full template sequence is stripped first."""
        ids = [int(t) for t in token_ids]
        if ids and ids[0] == QWEN_IM_START:
            ids = strip_template_prefix(ids)
        if tuple(ids[-len(KREA2_TAIL):]) != KREA2_TAIL:
            raise ValueError(f"Krea 2 DiT tokens must end with the 5-token tail {KREA2_TAIL}, got {ids[-5:]}")
        return TokenMap.from_lengths(ids, suffix=len(KREA2_TAIL), targets=dict(targets or {}),
                                     branch_text_len={branch: len(ids)})

    def layout(self, n_img: int, token_map: TokenMap) -> SeqLayout:
        return SeqLayout("txt_img", n_img, token_map.text_len)

    def output_projection(self, block: int) -> Callable[[Tensor], Tensor]:
        self._require_model()
        return self.model.blocks[block].attn.wo

    # ------------------------------------------------------------------ binding
    def _check_model(self, dm: nn.Module) -> None:
        blocks = getattr(dm, "blocks", None)
        if blocks is None or not hasattr(dm, "txtfusion") or not hasattr(dm, "txtmlp"):
            raise TypeError(f"{type(dm).__name__} is not a Krea 2 DiT (needs .blocks/.txtfusion/.txtmlp)")
        if len(blocks) != KREA2_CAPABILITY.num_blocks:
            raise ValueError(f"expected {KREA2_CAPABILITY.num_blocks} blocks, found {len(blocks)}")
        attn = blocks[0].attn
        if (attn.heads, attn.kvheads, attn.headdim) != (KREA2_CAPABILITY.num_q_heads, KREA2_CAPABILITY.num_kv_heads,
                                                        KREA2_CAPABILITY.head_dim):
            raise ValueError(f"unexpected attention geometry {(attn.heads, attn.kvheads, attn.headdim)}")

    def _build_patches(self, session: XAISession) -> dict[str, Callable[..., Any]]:
        patches: dict[str, Callable[..., Any]] = {
            "txtfusion.forward": self._txtfusion_forward(session),
            "txtmlp.forward": self._txtmlp_forward(session),
        }
        for i, blk in enumerate(self.model.blocks):
            patches[f"blocks.{i}.forward"] = self._block_forward(i, blk, session)
        return patches

    # ------------------------------------------------------------------ pre-DiT hooks
    def _txtfusion_forward(self, session: XAISession) -> Callable[..., Tensor]:
        orig = original_forward(self.model.txtfusion)

        def forward(x: Tensor, mask: Optional[Tensor] = None, transformer_options: dict = {}) -> Tensor:
            self.fwd.reset(transformer_options)
            ctx = maybe_step_context(transformer_options, x.shape[0]) if session.enabled else None
            self.fwd.extra["ctx"] = ctx
            return orig(span_swap(session, "text_in", x, ctx), mask=mask, transformer_options=transformer_options)

        return forward

    def _txtmlp_forward(self, session: XAISession) -> Callable[..., Tensor]:
        orig = original_forward(self.model.txtmlp)

        def forward(x: Tensor) -> Tensor:
            return span_swap(session, "txt_embed", orig(x), self.fwd.extra.get("ctx"))

        return forward

    # ------------------------------------------------------------------ block hooks
    def _block_forward(self, i: int, blk: nn.Module, session: XAISession) -> Callable[..., Tensor]:
        orig = original_forward(blk)
        ns = model_namespace(blk.attn)

        def forward(x: Tensor, vec: Tensor, freqs: Optional[Tensor], mask: Optional[Tensor] = None,
                    timestep_zero_index: Optional[int] = None, transformer_options: dict = {}) -> Tensor:
            ctx = maybe_step_context(transformer_options, x.shape[0]) if session.enabled else None
            if ctx is None:
                self.block_calls[i] += 1
                return orig(x, vec, freqs, mask, timestep_zero_index=timestep_zero_index,
                            transformer_options=transformer_options)
            bypass = session.bypass_mode(i, ctx)
            if bypass == "full_block":
                self.bypass_calls[i] += 1
                return x
            self.block_calls[i] += 1
            if not session_touches(session, i, ctx):
                return orig(x, vec, freqs, mask, timestep_zero_index=timestep_zero_index,
                            transformer_options=transformer_options)
            if timestep_zero_index is not None or len(transformer_options.get("reference_image_num_tokens", ())):
                raise NotImplementedError("XAI hooks do not support Krea 2 reference latents")
            return self._block_slow(i, blk, ns, session, ctx, bypass, x, vec, freqs, mask, transformer_options)

        return forward

    def _block_slow(self, i: int, blk: nn.Module, ns: Mapping[str, Any], session: XAISession, ctx: StepContext,
                    bypass: Optional[str], x: Tensor, vec: Tensor, freqs: Optional[Tensor], mask: Optional[Tensor],
                    options: dict) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = blk.mod(vec)
        n_txt = int(options["img_slice"][0])
        layout = SeqLayout("txt_img", x.shape[1] - n_txt, n_txt)
        concepts = self._concept_init(session, ctx, x.shape[0], x.shape[-1], x)

        if bypass != "attn_only":
            h = (1 + prescale) * blk.prenorm(x) + preshift
            attn_out, k, v, out = self._attention(i, blk.attn, ns, session, ctx, layout, h, freqs, mask, options, pregate)
            del h
            if concepts is not None:
                concepts = self._concept_attn(i, blk, session, ctx, layout, concepts, k, v, out,
                                              prescale, preshift, pregate)
            del k, v, out
            x = x + pregate * attn_out
            del attn_out
        if bypass != "mlp_only":
            x = x + postgate * blk.mlp((1 + postscale) * blk.postnorm(x) + postshift)
            if concepts is not None:
                concepts = concepts + postgate * blk.mlp((1 + postscale) * blk.postnorm(concepts) + postshift)
        if concepts is not None:
            self.fwd.concepts = concepts
        session.record("residual_out", x, block=i, ctx=ctx)
        return x

    def _attention(self, i: int, attn: nn.Module, ns: Mapping[str, Any], session: XAISession, ctx: StepContext,
                   layout: SeqLayout, h: Tensor, freqs: Optional[Tensor], mask: Optional[Tensor], options: dict,
                   g1: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Mirror of ``krea2.model.Attention.forward`` with the kernel routed through the session.

        Returns ``(attn_out [B, L, F], k [B, 12, L, d] post-RoPE pre-repeat, v [B, 12, L, d],
        ctx [B, L, 48*d])``.
        """
        patches = options.get("patches", {})
        extra_options = options.copy()
        B, L, _ = h.shape
        Hq, Hkv, D = attn.heads, attn.kvheads, attn.headdim
        q, k, v, gate_logits = attn.wq(h), attn.wk(h), attn.wv(h), attn.gate(h)
        q = q.view(B, L, Hq, D).transpose(1, 2)
        k = k.view(B, L, Hkv, D).transpose(1, 2)
        v = v.view(B, L, Hkv, D).transpose(1, 2)
        q, k = attn.qknorm(q, k)
        if "block_index" in options and "attn1_patch" in patches:
            for p in patches["attn1_patch"]:
                o = p(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra_options)
                q, k, v = o.get("q", q), o.get("k", k), o.get("v", v)
                freqs, mask = o.get("pe", freqs), o.get("attn_mask", mask)
        k, v = kv_patch(session, i, ctx, k, v, layout)  # 12 KV-head groups, pre-RoPE
        if freqs is not None:
            q, k = ns["apply_rope"](q, k, freqs)

        gate_value = F.sigmoid(gate_logits)
        gate_value = gate_override(gate_value, active_specs(session, "gate_override", i, ctx), ctx, layout)
        AT, optimized_attention_masked = ns["AttentionTensorContainer"], ns["optimized_attention_masked"]

        def native(q_: Tensor, k_: Tensor, v_: Tensor) -> Tensor:
            if Hkv != Hq:
                k_, v_ = k_.repeat_interleave(Hq // Hkv, dim=1), v_.repeat_interleave(Hq // Hkv, dim=1)
            out = optimized_attention_masked(AT(q_), AT(k_), AT(v_), Hq, mask=mask, skip_reshape=True,
                                             preferred_attention=attn.comfy_attention, transformer_options=options)
            return out.reshape(B, L, Hq, D).transpose(1, 2)

        o = session.attention(q, k, v, block=i, ctx=ctx, layout=layout, native=native, attn_mask=mask,
                              extras={"gate_logits": gate_logits, "g1": g1})
        out = o.transpose(1, 2).reshape(B, L, Hq * D)
        if "block_index" in options and "attn1_output_patch" in patches:
            for p in patches["attn1_output_patch"]:
                out = p(out, extra_options)
        session.record("attn_pre_proj", out, block=i, ctx=ctx)
        session.record("gate_logits", gate_logits, block=i, ctx=ctx)
        session.record("gate_value", gate_value, block=i, ctx=ctx)
        attn_out = attn.wo(out * gate_value)
        session.record("attn_out", attn_out, block=i, ctx=ctx)
        return attn_out, k, v, out

    def _concept_attn(self, i: int, blk: nn.Module, session: XAISession, ctx: StepContext, layout: SeqLayout,
                      c: Tensor, k_x: Tensor, v_x: Tensor, o_x: Tensor, prescale: Tensor, preshift: Tensor,
                      pregate: Tensor) -> Tensor:
        """ConceptAttention stream through this block's attention (own gate, ``wo`` and ``g1``).

        Concepts attend to ``[txt; img; concepts]`` (GQA, no RoPE: text-like zero positions);
        main-stream tokens never see them. Sinks receive ``o_c`` (pre-gate ctx) and
        ``gate_logits_c`` so ``pre_gate`` vs ``post_gate`` readouts can be formed.
        """
        attn = blk.attn
        B, R, _ = c.shape
        Hq, Hkv, D = attn.heads, attn.kvheads, attn.headdim
        hc = (1 + prescale) * blk.prenorm(c) + preshift
        qc, kc, vc, gc = attn.wq(hc), attn.wk(hc), attn.wv(hc), attn.gate(hc)
        qc, kc = attn.qknorm(qc.view(B, R, Hq, D).transpose(1, 2), kc.view(B, R, Hkv, D).transpose(1, 2))
        vc = vc.view(B, R, Hkv, D).transpose(1, 2)
        o_c = concept_stream_attention(qc, k_x, v_x, kc, vc).transpose(1, 2).reshape(B, R, Hq * D)
        self._emit_concepts(session, i, ctx, o_c, o_x, layout, gate_logits_c=gc)
        return c + pregate * attn.wo(o_c * F.sigmoid(gc))
