"""Anima adapter (cross-attention DiT, ``comfy/ldm/cosmos/predict2.py`` + ``comfy/ldm/anima/model.py``).

Hook points (per block ``i`` of ``MiniTrainDIT.blocks``, 28 blocks, 16 heads x 128):

========================  =====================================================================
``blocks.i.forward``      step/CFG context, ``true_block_bypass`` (``full_block`` returns the
                          input residual; block body never runs), ``residual_out`` tap, ``g1``
``blocks.i.cross_attn``   image→text attention routed through :meth:`XAISession.attention`
                          (``q/k/v`` after per-head RMSNorm, no RoPE), ``kv_patch`` on text
                          K/V, ``attn_pre_proj`` (before ``output_proj``), ``attn_out`` (after
                          ``output_proj``, before the ``gate_cross_attn`` modulation), ``ctx`` swap
``blocks.i.self_attn``    ``attn_only`` bypass (image self-attention contributes 0)
``blocks.i.mlp``          ``mlp_only`` bypass
========================  =====================================================================

Text slots are the T5 token positions of the LLMAdapter output ``ctx [B, L, 1024]`` with
``L = max(512, Lt)``: ``S=[0, Lt-1)`` prompt, ``R=[Lt-1, Lt)`` (``</s>``), ``P=[Lt, L)`` zero pad.
The text stream is static (no per-block update), so ``concept_stream`` and ``gate_override``
are structurally N/A; ``pre_dit_span_swap`` acts on ``point="ctx"``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from ..core.session import XAISession
from ._comfy_hooks import (
    BlockState,
    ComfyDiTAdapter,
    kv_patch,
    maybe_step_context,
    observers_active,
    original_forward,
    session_touches,
    span_swap,
)
from .base import AdapterCapability, Branch, SeqLayout, TokenMap

Tensor = torch.Tensor

ANIMA_CAPABILITY = AdapterCapability(
    name="anima", attention_kind="cross_attn", seq_order="cross", num_blocks=28, num_q_heads=16,
    num_kv_heads=16, head_dim=128, text_updated_per_block=False, supports_concept_attention=False,
    has_output_gate=False, text_pad_to=512,
)

T5_EOS_ID = 1  # ``</s>``
T5_PAD_ID = 0


class AnimaAdapter(ComfyDiTAdapter):
    UNSUPPORTED_HOOKS = frozenset({"concept_stream", "gate_override"})
    SWAP_POINTS = frozenset({"ctx"})
    BYPASS_MODES = frozenset({"full_block", "attn_only", "cross_attn_only", "mlp_only"})

    @property
    def capability(self) -> AdapterCapability:
        return ANIMA_CAPABILITY

    # ------------------------------------------------------------------ tokens
    @staticmethod
    def dit_token_ids(tokenized: Mapping[str, Any]) -> list[int]:
        """T5 slot IDs from ``clip.tokenize(prompt)`` (``tokenized["t5xxl"][0]`` pairs)."""
        return [int(t[0]) for t in tokenized["t5xxl"][0]]

    def build_token_map(self, token_ids: Sequence[int], *, branch: Branch = "cond",
                        targets: Optional[Mapping[str, Any]] = None) -> TokenMap:
        """``token_ids``: T5 IDs ending in ``</s>`` (trailing zero padding, if given, is ignored)."""
        ids = [int(t) for t in token_ids]
        while ids and ids[-1] == T5_PAD_ID:
            ids.pop()
        if not ids or ids[-1] != T5_EOS_ID:
            raise ValueError("Anima T5 token IDs must end with </s> (id 1)")
        lt = len(ids)
        total = max(ANIMA_CAPABILITY.text_pad_to or 0, lt)
        return TokenMap.from_lengths(ids + [T5_PAD_ID] * (total - lt), suffix=1, pad=total - lt,
                                     targets=dict(targets or {}), branch_text_len={branch: lt})

    def layout(self, n_img: int, token_map: TokenMap) -> SeqLayout:
        return SeqLayout("cross", n_img, token_map.text_len)

    def output_projection(self, block: int) -> Callable[[Tensor], Tensor]:
        self._require_model()
        return self.model.blocks[block].cross_attn.output_proj

    # ------------------------------------------------------------------ binding
    def _check_model(self, dm: nn.Module) -> None:
        blocks = getattr(dm, "blocks", None)
        if blocks is None or not hasattr(dm, "llm_adapter"):
            raise TypeError(f"{type(dm).__name__} is not an Anima DiT (needs .blocks and .llm_adapter)")
        if len(blocks) != ANIMA_CAPABILITY.num_blocks:
            raise ValueError(f"expected {ANIMA_CAPABILITY.num_blocks} blocks, found {len(blocks)}")

    def _build_patches(self, session: XAISession) -> dict[str, Callable[..., Any]]:
        patches: dict[str, Callable[..., Any]] = {}
        for i, blk in enumerate(self.model.blocks):
            patches[f"blocks.{i}.forward"] = self._block_forward(i, blk, session)
            patches[f"blocks.{i}.cross_attn.forward"] = self._cross_attn_forward(i, blk.cross_attn, session)
            patches[f"blocks.{i}.self_attn.forward"] = self._zero_if_bypassed(i, blk.self_attn, "attn_only")
            patches[f"blocks.{i}.mlp.forward"] = self._zero_if_bypassed(i, blk.mlp, "mlp_only")
        return patches

    # ------------------------------------------------------------------ hooks
    def _block_forward(self, i: int, blk: nn.Module, session: XAISession) -> Callable[..., Tensor]:
        orig = original_forward(blk)

        def forward(x_B_T_H_W_D: Tensor, emb_B_T_D: Tensor, crossattn_emb: Tensor,
                    rope_emb_L_1_1_D: Optional[Tensor] = None, adaln_lora_B_T_3D: Optional[Tensor] = None,
                    extra_per_block_pos_emb: Optional[Tensor] = None, transformer_options: Optional[dict] = {}) -> Tensor:
            args = (x_B_T_H_W_D, emb_B_T_D, crossattn_emb)
            kwargs = dict(rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                          extra_per_block_pos_emb=extra_per_block_pos_emb, transformer_options=transformer_options)
            ctx = maybe_step_context(transformer_options, x_B_T_H_W_D.shape[0]) if session.enabled else None
            if ctx is None:
                self.block_calls[i] += 1
                return orig(*args, **kwargs)
            bypass = session.bypass_mode(i, ctx)
            if bypass == "full_block":
                self.bypass_calls[i] += 1
                return x_B_T_H_W_D
            g1 = None
            if observers_active(session, i, ctx):  # e.g. Cont^(l) needs the post-Wo modulation
                mod = blk.adaln_modulation_cross_attn(emb_B_T_D)
                if blk.use_adaln_lora:
                    mod = mod + adaln_lora_B_T_3D
                g1 = mod.chunk(3, dim=-1)[2]
            self.fwd.block = BlockState(i, ctx, bypass, g1)
            self.block_calls[i] += 1
            try:
                out = orig(*args, **kwargs)
            finally:
                self.fwd.block = None
            if session.wants("residual_out", i, ctx):
                session.record("residual_out", out.reshape(out.shape[0], -1, out.shape[-1]), block=i, ctx=ctx)
            return out

        return forward

    def _zero_if_bypassed(self, i: int, module: nn.Module, mode: str) -> Callable[..., Tensor]:
        orig = original_forward(module)

        def forward(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            st = self.fwd.block
            if st is not None and st.block == i and st.bypass == mode:
                return torch.zeros_like(x)  # residual += gate * 0: sub-layer removed, never called
            return orig(x, *args, **kwargs)

        return forward

    def _cross_attn_forward(self, i: int, attn: nn.Module, session: XAISession) -> Callable[..., Tensor]:
        orig = original_forward(attn)

        def forward(x: Tensor, context: Optional[Tensor] = None, rope_emb: Optional[Tensor] = None,
                    transformer_options: Optional[dict] = {}) -> Tensor:
            st = self.fwd.block
            if st is None or st.block != i:
                return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            ctx = st.ctx
            if st.bypass == "cross_attn_only":
                return torch.zeros_like(x)
            context = span_swap(session, "ctx", context, ctx)
            if not session_touches(session, i, ctx):
                return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

            q, k, v = attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            B, S, H, D = q.shape
            qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, N, D]
            layout = SeqLayout("cross", S, kh.shape[2])
            kh, vh = kv_patch(session, i, ctx, kh, vh, layout)

            def native(q_: Tensor, k_: Tensor, v_: Tensor) -> Tensor:
                out = attn.attn_op(q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2),
                                   transformer_options=transformer_options, preferred_attention=attn.comfy_attention)
                return out.reshape(B, S, H, D).transpose(1, 2)

            extras = {} if st.g1 is None else {"g1": st.g1}
            o = session.attention(qh, kh, vh, block=i, ctx=ctx, layout=layout, native=native, extras=extras)
            o = o.transpose(1, 2).reshape(B, S, H * D)
            session.record("attn_pre_proj", o, block=i, ctx=ctx)
            out = attn.output_dropout(attn.output_proj(o))
            session.record("attn_out", out, block=i, ctx=ctx)
            return out

        return forward
