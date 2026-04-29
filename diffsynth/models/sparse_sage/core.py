"""
Sparse SageAttention — adapted from https://github.com/jt-zhang/Sparse_SageAttention_API
via https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast

Licensed under the Apache License, Version 2.0.
"""

import torch
from .quant_per_block import per_block_int8
from .sparse_int8_attn import forward as sparse_sageattn_fwd


def sparse_sageattn(q, k, v, mask_id=None, is_causal=False, tensor_layout="HND"):
    """INT8 block-sparse attention.

    Args:
        q, k, v: [B, H, S, D] (HND layout) or [B, S, H, D] (NHD layout)
        mask_id: int8 tensor [B, H, ceil(S_q/128), ceil(S_k/64)], 1=attend, 0=skip
        tensor_layout: "HND" or "NHD"
    """
    if mask_id is None:
        # dense fallback
        mask_id = torch.ones(
            (q.shape[0], q.shape[1], (q.shape[2] + 128 - 1) // 128, (q.shape[2] + 64 - 1) // 64),
            dtype=torch.int8, device=q.device,
        )

    output_dtype = q.dtype
    if output_dtype in (torch.bfloat16, torch.float32):
        v = v.to(torch.float16)

    seq_dim = 1 if tensor_layout == "NHD" else 2
    km = k.mean(dim=seq_dim, keepdim=True)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, tensor_layout=tensor_layout)

    return sparse_sageattn_fwd(
        q_int8, k_int8, mask_id, v, q_scale, k_scale,
        is_causal=is_causal, tensor_layout=tensor_layout, output_dtype=output_dtype,
    )
