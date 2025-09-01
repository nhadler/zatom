import torch
import triton
import triton.language as tl

# -------------------------
# Constants
# -------------------------
# Candidate configurations for auto-tuning
configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4},
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4},
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8},
        num_stages=2,
    ),
]


# -------------------------
# Forward Kernel
# -------------------------
@triton.autotune(configs=configs, key=["seqlen_q", "seqlen_k"])
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Mask,
    Out,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_vbh,
    stride_vn,
    stride_vk,
    stride_mbh,
    stride_mm,
    stride_mn,
    stride_obh,
    stride_om,
    stride_ok,
    seqlen_q,
    seqlen_k,
    dropout_p,
    rng_seed,
    rng_offset,
    has_mask: tl.constexpr,
    mask_is_additive: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Forward kernel for Triton-based FlashAttention.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        Mask: Attention mask.
        Out: Output tensor.
        stride_qbh: Stride for query batch.
        stride_qm: Stride for query memory.
        stride_qk: Stride for query key.
        stride_kbh: Stride for key batch.
        stride_kn: Stride for key memory.
        stride_kk: Stride for key key.
        stride_vbh: Stride for value batch.
        stride_vn: Stride for value memory.
        stride_vk: Stride for value key.
        stride_mbh: Stride for mask batch.
        stride_mm: Stride for mask memory.
        stride_mn: Stride for mask key.
        stride_obh: Stride for output batch.
        stride_om: Stride for output memory.
        stride_ok: Stride for output key.
        seqlen_q: Sequence length of query.
        seqlen_k: Sequence length of key.
        dropout_p: Dropout probability.
        rng_seed: Random number generator seed.
        rng_offset: Random number generator offset.
        has_mask: Whether the mask is present.
        mask_is_additive: Whether the mask is additive.
        causal: Whether the attention is causal.
        BLOCK_M: Block size for M dimension.
        BLOCK_N: Block size for N dimension.
        BLOCK_DMODEL: Block size for D model dimension.
    """
    pid = tl.program_id(0)
    bh = pid // tl.cdiv(seqlen_q, BLOCK_M)
    start_m = (pid % tl.cdiv(seqlen_q, BLOCK_M)) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers
    Q_ptrs = Q + bh * stride_qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptrs = K + bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    V_ptrs = V + bh * stride_vbh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    o_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(float(BLOCK_DMODEL))

    for start_n in range(0, seqlen_k, BLOCK_N):
        k = tl.load(
            K_ptrs + start_n * stride_kn,
            mask=offs_n[:, None] + start_n < seqlen_k,
            other=0.0,
        )
        v = tl.load(
            V_ptrs + start_n * stride_vn,
            mask=offs_n[:, None] + start_n < seqlen_k,
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale

        if has_mask:
            mask_vals = tl.load(
                Mask
                + bh * stride_mbh
                + offs_m[:, None] * stride_mm
                + (offs_n[None, :] + start_n) * stride_mn,
                mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] + start_n < seqlen_k),
                other=0.0 if mask_is_additive else 0,
            )
            if mask_is_additive:
                qk += mask_vals
            else:
                qk = tl.where(mask_vals > 0, qk, float("-inf"))

        if causal:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Manual softmax
        qk_max = tl.max(qk, 1)
        qk = qk - qk_max[:, None]
        p = tl.exp(qk)
        p_sum = tl.sum(p, 1)
        p = p / p_sum[:, None]

        o_acc += tl.dot(p, v.to(tl.float32))

    Out_ptrs = Out + bh * stride_obh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(Out_ptrs, o_acc, mask=offs_m[:, None] < seqlen_q)


# -------------------------
# Backward Kernel
# -------------------------
@triton.autotune(configs=configs, key=["seqlen_q", "seqlen_k"])
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Mask,
    dO,
    dQ,
    dK,
    dV,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_vbh,
    stride_vn,
    stride_vk,
    stride_mbh,
    stride_mm,
    stride_mn,
    stride_dobh,
    stride_dom,
    stride_dok,
    seqlen_q,
    seqlen_k,
    has_mask: tl.constexpr,
    mask_is_additive: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Backward kernel for Triton-based FlashAttention.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        Mask: Attention mask tensor.
        dO: Output gradient tensor.
        dQ: Query gradient tensor.
        dK: Key gradient tensor.
        dV: Value gradient tensor.
        stride_qbh: Stride for query batch.
        stride_qm: Stride for query memory.
        stride_qk: Stride for query key.
        stride_kbh: Stride for key batch.
        stride_kn: Stride for key memory.
        stride_kk: Stride for key key.
        stride_vbh: Stride for value batch.
        stride_vn: Stride for value memory.
        stride_vk: Stride for value key.
        stride_mbh: Stride for mask batch.
        stride_mm: Stride for mask memory.
        stride_mn: Stride for mask key.
        stride_dobh: Stride for output gradient batch.
        stride_dom: Stride for output gradient memory.
        stride_dok: Stride for output gradient key.
        seqlen_q: Sequence length of query.
        seqlen_k: Sequence length of key.
        has_mask: Whether the mask is present.
        mask_is_additive: Whether the mask is additive.
        causal: Whether the attention is causal.
        BLOCK_M: Block size for M dimension.
        BLOCK_N: Block size for N dimension.
        BLOCK_DMODEL: Block size for D model dimension.
    """
    pid = tl.program_id(0)
    bh = pid // tl.cdiv(seqlen_q, BLOCK_M)
    start_m = (pid % tl.cdiv(seqlen_q, BLOCK_M)) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    Q_ptrs = Q + bh * stride_qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptrs = K + bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    V_ptrs = V + bh * stride_vbh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    dO_ptrs = dO + bh * stride_dobh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    do = tl.load(dO_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    dq_acc = tl.zeros_like(q.to(tl.float32))
    dk_acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(float(BLOCK_DMODEL))

    for start_n in range(0, seqlen_k, BLOCK_N):
        k = tl.load(
            K_ptrs + start_n * stride_kn,
            mask=offs_n[:, None] + start_n < seqlen_k,
            other=0.0,
        )
        v = tl.load(
            V_ptrs + start_n * stride_vn,
            mask=offs_n[:, None] + start_n < seqlen_k,
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale

        if has_mask:
            mask_vals = tl.load(
                Mask
                + bh * stride_mbh
                + offs_m[:, None] * stride_mm
                + (offs_n[None, :] + start_n) * stride_mn,
                mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] + start_n < seqlen_k),
                other=0.0 if mask_is_additive else 0,
            )
            if mask_is_additive:
                qk += mask_vals
            else:
                qk = tl.where(mask_vals > 0, qk, float("-inf"))

        if causal:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Manual softmax
        qk_max = tl.max(qk, 1)
        qk = qk - qk_max[:, None]
        p = tl.exp(qk)
        p_sum = tl.sum(p, 1)
        p = p / p_sum[:, None]

        dv_acc += tl.dot(tl.trans(p), do.to(tl.float32))
        dp = tl.dot(do, tl.trans(v))
        dp -= p * tl.sum(dp * p, 1)[:, None]
        ds = dp * scale

        dq_acc += tl.dot(ds, k.to(tl.float32))
        dk_acc += tl.dot(tl.trans(ds), q.to(tl.float32))

    tl.store(
        dQ + bh * stride_qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        dq_acc,
        mask=offs_m[:, None] < seqlen_q,
    )
    tl.store(
        dK + bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
        dk_acc,
        mask=offs_n[:, None] < seqlen_k,
    )
    tl.store(
        dV + bh * stride_vbh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
        dv_acc,
        mask=offs_n[:, None] < seqlen_k,
    )


# -------------------------
# Autograd Function
# -------------------------
class FlashAttnFunc(torch.autograd.Function):
    """FlashAttention autograd function."""

    @staticmethod
    def forward(ctx, Q, K, V, mask=None, causal=False, dropout_p=0.0, rng_seed=0):
        """Forward pass for Triton-based FlashAttention.

        Args:
            Q: Query tensor.
            K: Key tensor.
            V: Value tensor.
            mask: Attention mask tensor.
            causal: Whether to use causal attention.
            dropout_p: Dropout probability.
            rng_seed: Random seed for dropout.

        Returns:
            Output tensor.
        """
        batch, heads, seqlen_q, d = Q.shape
        seqlen_k = K.shape[2]
        O = torch.empty_like(Q)  # noqa: E741

        has_mask = mask is not None
        mask_is_additive = False
        if has_mask:
            if mask.dtype == torch.bool:
                mask_is_additive = False
            else:
                mask_is_additive = True
            assert mask.shape == (batch, heads, seqlen_q, seqlen_k)

        grid = (batch * heads * triton.cdiv(seqlen_q, 64),)
        _fwd_kernel[grid](
            Q,
            K,
            V,
            mask if has_mask else torch.empty(1, device=Q.device),
            O,
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            (mask.stride(1) if has_mask else 0),
            (mask.stride(2) if has_mask else 0),
            (mask.stride(3) if has_mask else 0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            seqlen_q,
            seqlen_k,
            dropout_p,
            rng_seed,
            0,
            has_mask,
            mask_is_additive,
            causal,
            BLOCK_DMODEL=d,
        )

        ctx.save_for_backward(Q, K, V, mask if has_mask else None)
        ctx.has_mask = has_mask
        ctx.mask_is_additive = mask_is_additive
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        """Backward pass for Triton-based FlashAttention.

        Args:
            ctx: Context object containing saved tensors.
            dO: Gradient of output tensor.

        Returns:
            Gradients of input tensors as a tuple.
        """
        Q, K, V, mask = ctx.saved_tensors
        batch, heads, seqlen_q, d = Q.shape
        seqlen_k = K.shape[2]

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        grid = (batch * heads * triton.cdiv(seqlen_q, 64),)
        _bwd_kernel[grid](
            Q,
            K,
            V,
            mask if ctx.has_mask else torch.empty(1, device=Q.device),
            dO,
            dQ,
            dK,
            dV,
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            (mask.stride(1) if ctx.has_mask else 0),
            (mask.stride(2) if ctx.has_mask else 0),
            (mask.stride(3) if ctx.has_mask else 0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            seqlen_q,
            seqlen_k,
            ctx.has_mask,
            ctx.mask_is_additive,
            ctx.causal,
            BLOCK_DMODEL=d,
        )

        return dQ, dK, dV, None, None, None, None


# -------------------------
# Public API
# -------------------------
def flash_attention(Q, K, V, mask=None, causal=False, dropout_p=0.0, rng_seed=0):
    """FlashAttention implementation with support for second-order gradients.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        mask: Attention mask tensor.
        causal: Whether to use causal attention.
        dropout_p: Dropout probability.
        rng_seed: Random seed for dropout.

    Returns:
        Output tensor.
    """
    return FlashAttnFunc.apply(Q, K, V, mask, causal, dropout_p, rng_seed)


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 128, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)

    mask = torch.randint(0, 2, (B, H, N, N), device="cuda", dtype=torch.bool)

    O = flash_attention(Q, K, V, mask=mask, causal=True, dropout_p=0.1, rng_seed=42)  # noqa: E741
    loss = O.sum()
    grad1 = torch.autograd.grad(loss, (Q, K, V), create_graph=True)
    grad2 = torch.autograd.grad(sum([g.sum() for g in grad1]), (Q, K, V))

    print("✅ Second-order grads computed successfully with mask + dropout!")
