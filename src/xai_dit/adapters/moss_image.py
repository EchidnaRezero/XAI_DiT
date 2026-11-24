"""MOSS-Image adapter (single-stream joint-attention DiT, ``comfy/ldm/moss_image/model.py``).

Sequence ``[img N | txt L]`` through 42 blocks (24 heads x 128, no GQA). Hook points:

==========================  ===================================================================
``text_embedder.forward``   pre-DiT: ``te.out`` swap (refiner input ``[B, L, 4096]``) and
                            ``refiner.out`` swap (DiT text tokens ``[B, L, 3072]``); records
                            the text length and resets per-forward state
``layers.i.forward``        block slow path (only when the session touches ``(i, step)``):
                            ``kv_patch`` on pre-norm/pre-RoPE K/V (``attn.qkv_pre``), attention
                            via :meth:`XAISession.attention` on post-norm/RoPE ``q/k/v``,
                            ``attn_pre_proj`` (= ``attn.out``, before ``proj["o"]``),
                            ``attn_out`` (= ``attn.proj``, before ``gate1``), ``residual_out``,
                            ``true_block_bypass``, ``concept_stream``
==========================  ===================================================================

Token spans (template included, byte-level BPE): ``R_prefix`` = ``<|im_start|>user\\n``,
``S`` = prompt, ``R`` (alias target ``R_suffix``) = ``<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``,
validated against the real Qwen3.5 special-token IDs. No padding.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from ..core.heatmap import concept_stream_attention
from ..core.session import XAISession
from ._comfy_hooks import (
    ComfyDiTAdapter,
    kv_patch,
    maybe_step_context,
    model_namespace,
    original_forward,
    session_touches,
    span_swap,
)
from .base import AdapterCapability, Branch, SeqLayout, StepContext, TokenMap

Tensor = torch.Tensor

MOSS_IMAGE_CAPABILITY = AdapterCapability(
    name="moss_image", attention_kind="joint_attn", seq_order="img_txt", num_blocks=42, num_q_heads=24,
    num_kv_heads=24, head_dim=128, text_updated_per_block=True, supports_concept_attention=True,
    has_output_gate=False, text_pad_to=None,
)

# Qwen3.5 tokenizer (``comfy/text_encoders/qwen35_tokenizer/tokenizer_config.json``, ``vocab.json``).
QWEN35_IM_START = 248045
QWEN35_IM_END = 248046
QWEN35_THINK = 248068
QWEN35_NEWLINE = 198


class MOSSImageAdapter(ComfyDiTAdapter):
    UNSUPPORTED_HOOKS = frozenset({"gate_override"})
    SWAP_POINTS = frozenset({"te.out", "refiner.out"})

    def __init__(self, diffusion_model: Any = None, *, im_start: int = QWEN35_IM_START,
                 im_end: int = QWEN35_IM_END, think: int = QWEN35_THINK, newline: int = QWEN35_NEWLINE) -> None:
        super().__init__(diffusion_model)
        self.im_start, self.im_end, self.think, self.newline = im_start, im_end, think, newline

    @property
    def capability(self) -> AdapterCapability:
        return MOSS_IMAGE_CAPABILITY

    # ------------------------------------------------------------------ tokens
    @staticmethod
    def dit_token_ids(tokenized: Mapping[str, Any]) -> list[int]:
        """All template-wrapped IDs from ``clip.tokenize(prompt)`` (the DiT keeps every token)."""
        return [int(t[0]) for t in tokenized["qwen35_9b"][0]]

    def build_token_map(self, token_ids: Sequence[int], *, branch: Branch = "cond",
                        targets: Optional[Mapping[str, Any]] = None) -> TokenMap:
        ids = [int(t) for t in token_ids]
        try:
            start = ids.index(self.im_start)
            prefix = ids.index(self.newline, start + 1) + 1  # "<|im_start|>" "user" "\n"
        except ValueError as e:
            raise ValueError("MOSS-Image tokens must start with the '<|im_start|>user\\n' template") from e
        if start != 0:
            raise ValueError(f"unexpected {start} tokens before <|im_start|>")
        ends = [j for j, t in enumerate(ids) if t == self.im_end]
        if not ends or ends[-1] < prefix:
            raise ValueError("MOSS-Image tokens lack the trailing '<|im_end|>' template")
        suffix_ids = ids[ends[-1]:]
        if self.im_start not in suffix_ids or self.think not in suffix_ids:
            raise ValueError(f"suffix template {suffix_ids} lacks <|im_start|>assistant / <think>")
        tm = TokenMap.from_lengths(ids, prefix=prefix, suffix=len(suffix_ids), targets=dict(targets or {}),
                                   branch_text_len={branch: len(ids)})
        return TokenMap(tm.token_ids, tm.spans, {**tm.targets, "R_suffix": tm.spans["R"]},
                        tm.branch_text_len, tm.occurrence_idx)

    def layout(self, n_img: int, token_map: TokenMap) -> SeqLayout:
        return SeqLayout("img_txt", n_img, token_map.text_len)

    def output_projection(self, block: int) -> Callable[[Tensor], Tensor]:
        self._require_model()
        return self.model.layers[block].proj["o"]

    # ------------------------------------------------------------------ binding
    def _check_model(self, dm: nn.Module) -> None:
        layers = getattr(dm, "layers", None)
        if layers is None or not hasattr(dm, "text_embedder"):
            raise TypeError(f"{type(dm).__name__} is not a MOSS-Image DiT (needs .layers and .text_embedder)")
        if len(layers) != MOSS_IMAGE_CAPABILITY.num_blocks:
            raise ValueError(f"expected {MOSS_IMAGE_CAPABILITY.num_blocks} blocks, found {len(layers)}")

    def _build_patches(self, session: XAISession) -> dict[str, Callable[..., Any]]:
        patches: dict[str, Callable[..., Any]] = {"text_embedder.forward": self._text_forward(session)}
        for i, blk in enumerate(self.model.layers):
            patches[f"layers.{i}.forward"] = self._block_forward(i, blk, session)
        return patches

    # ------------------------------------------------------------------ hooks
    def _text_forward(self, session: XAISession) -> Callable[..., Tensor]:
        te = self.model.text_embedder
        orig = original_forward(te)

        def forward(x: Tensor, pe: Tensor, transformer_options: dict = {}, mask: Optional[Tensor] = None) -> Tensor:
            self.fwd.reset(transformer_options)
            self.fwd.text_len = x.shape[1]
            ctx = maybe_step_context(transformer_options, x.shape[0]) if session.enabled else None
            x = span_swap(session, "te.out", x, ctx)
            out = orig(x, pe, transformer_options, mask)
            return span_swap(session, "refiner.out", out, ctx)

        return forward

    def _block_forward(self, i: int, blk: nn.Module, session: XAISession) -> Callable[..., Tensor]:
        orig = original_forward(blk)
        ns = model_namespace(blk)

        def forward(x: Tensor, mod: Sequence[Tensor], pe: Tensor, transformer_options: dict = {}, ref: Any = None,
                    ref_kv: Optional[Tensor] = None, kv_out: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
            ctx = maybe_step_context(transformer_options, x.shape[0]) if session.enabled else None
            if ctx is None:
                self.block_calls[i] += 1
                return orig(x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
            bypass = session.bypass_mode(i, ctx)
            if bypass == "full_block":
                self.bypass_calls[i] += 1
                return x
            self.block_calls[i] += 1
            if not session_touches(session, i, ctx):
                return orig(x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
            if ref is not None or ref_kv is not None or kv_out is not None:
                raise NotImplementedError("XAI hooks do not support MOSS-Image reference latents")
            if self.fwd.text_len is None:
                raise RuntimeError("text length unknown: text_embedder hook did not run for this forward")
            return self._block_slow(i, blk, ns, session, ctx, bypass, x, mod, pe, transformer_options, mask)

        return forward

    def _block_slow(self, i: int, blk: nn.Module, ns: Mapping[str, Any], session: XAISession, ctx: StepContext,
                    bypass: Optional[str], x: Tensor, mod: Sequence[Tensor], pe: Tensor, options: dict,
                    mask: Optional[Tensor]) -> Tensor:
        modulate, residual = ns["modulate"], ns["residual"]
        linear_input_act = ns["comfy"].ops.linear_input_act
        n_txt = self.fwd.text_len
        layout = SeqLayout("img_txt", x.shape[1] - n_txt, n_txt)
        concepts = self._concept_init(session, ctx, x.shape[0], x.shape[-1], x)

        if bypass != "attn_only":
            y = modulate(x, blk.norm1, mod, 0, None)
            q, k, v = self._qkv(blk, y)
            del y
            out, kh, vh = self._attention(i, blk, ns, session, ctx, layout, q, k, v, pe, options, mask, g1=mod[2])
            del q, k, v
            session.record("attn_pre_proj", out, block=i, ctx=ctx)
            attn_out = blk.proj["o"](out)
            session.record("attn_out", attn_out, block=i, ctx=ctx)
            if concepts is not None:
                concepts = self._concept_attn(i, blk, ns, session, ctx, layout, concepts, mod, kh, vh, out)
            del kh, vh, out
            x = residual(x, attn_out, mod, 2, None)
            del attn_out
        if bypass != "mlp_only":
            y = blk.proj["ffn_up_gate"](modulate(x, blk.norm2, mod, 3, None))
            x = residual(x, linear_input_act(blk.proj["ffn_down"], y, "situ"), mod, 5, None)
            del y
            if concepts is not None:
                yc = blk.proj["ffn_up_gate"](modulate(concepts, blk.norm2, mod, 3, None))
                concepts = torch.addcmul(concepts, linear_input_act(blk.proj["ffn_down"], yc, "situ"), mod[5])
        if concepts is not None:
            self.fwd.concepts = concepts
        session.record("residual_out", x, block=i, ctx=ctx)
        return x

    @staticmethod
    def _qkv(blk: nn.Module, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if blk.fused_qkv:
            return blk.proj["qkv"](y).chunk(3, dim=-1)
        return blk.proj["q"](y), blk.proj["k"](y), blk.proj["v"](y)

    def _attention(self, i: int, blk: nn.Module, ns: Mapping[str, Any], session: XAISession, ctx: StepContext,
                   layout: SeqLayout, q: Tensor, k: Tensor, v: Tensor, pe: Tensor, options: dict,
                   mask: Optional[Tensor], g1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Mirror of ``moss_image.model.attention`` with the kernel routed through the session.

        Returns ``(out [B, N, H*D], k [B, H, N, D] post-RoPE, v [B, H, N, D])``.
        """
        comfy = ns["comfy"]
        B, N, _ = q.shape
        H = blk.num_heads
        D = q.shape[-1] // H
        # kv_patch at attn.qkv_pre (pre-norm, pre-RoPE), per patching.KVCache contract
        if session.interventions_for("kv_patch", i, ctx):
            k4, v4 = kv_patch(session, i, ctx, k.view(B, N, H, D).transpose(1, 2), v.view(B, N, H, D).transpose(1, 2), layout)
            k, v = k4.transpose(1, 2).reshape(B, N, H * D), v4.transpose(1, 2).reshape(B, N, H * D)
        q, k = q.view(B, N, H, -1), k.view(B, N, H, -1)
        if comfy.model_management.in_training:
            rope_split_half = ns["rope_split_half"]
            q, k = rope_split_half(blk.q_norm(q), pe), rope_split_half(blk.k_norm(k), pe)
        else:
            q_scale, _, q_stream = comfy.ops.cast_bias_weight(blk.q_norm, q, offloadable=True)
            k_scale, _, k_stream = comfy.ops.cast_bias_weight(blk.k_norm, k, offloadable=True)
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, blk.q_norm.eps)
            comfy.ops.uncast_bias_weight(blk.q_norm, q_scale, None, q_stream)
            comfy.ops.uncast_bias_weight(blk.k_norm, k_scale, None, k_stream)
        q, k = q.flatten(2), k.flatten(2)
        qh, kh, vh = (t.view(B, N, H, D).transpose(1, 2) for t in (q, k, v))  # [B, H, N, D] views
        AT, optimized_attention = ns["AttentionTensorContainer"], ns["optimized_attention"]

        def native(q_: Tensor, k_: Tensor, v_: Tensor) -> Tensor:
            flat = [t.transpose(1, 2).reshape(B, N, H * D) for t in (q_, k_, v_)]
            out = optimized_attention(AT(flat[0]), AT(flat[1]), AT(flat[2]), H, mask=mask, transformer_options=options,
                                      preferred_attention=blk.comfy_attention)
            return out.reshape(B, N, H, D).transpose(1, 2)

        o = session.attention(qh, kh, vh, block=i, ctx=ctx, layout=layout, native=native, attn_mask=mask,
                              extras={"g1": g1})
        return o.transpose(1, 2).reshape(B, N, H * D), kh, vh

    def _concept_attn(self, i: int, blk: nn.Module, ns: Mapping[str, Any], session: XAISession, ctx: StepContext,
                      layout: SeqLayout, c: Tensor, mod: Sequence[Tensor], k_x: Tensor, v_x: Tensor,
                      o_x: Tensor) -> Tensor:
        """ConceptAttention stream: concepts attend to ``[img; txt; concepts]``, nothing attends to them.

        Concept tokens carry no RoPE (identity rotation); they share the block's modulation,
        projections, ``proj["o"]`` and ``gate1`` with the main stream.
        """
        B, R, _ = c.shape
        H = blk.num_heads
        yc = ns["modulate"](c, blk.norm1, mod, 0, None)
        qc, kc, vc = self._qkv(blk, yc)
        D = qc.shape[-1] // H
        qc = blk.q_norm(qc.reshape(B, R, H, D)).transpose(1, 2)
        kc = blk.k_norm(kc.reshape(B, R, H, D)).transpose(1, 2)
        vc = vc.reshape(B, R, H, D).transpose(1, 2)
        o_c = concept_stream_attention(qc, k_x, v_x, kc, vc).transpose(1, 2).reshape(B, R, H * D)
        self._emit_concepts(session, i, ctx, o_c, o_x, layout)
        return torch.addcmul(c, blk.proj["o"](o_c), mod[2])
