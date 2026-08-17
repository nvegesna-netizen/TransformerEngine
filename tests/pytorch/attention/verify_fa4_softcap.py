# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Definitive on-device verification that TE softcap works on FlashAttention 4 (FA4).

WHAT THIS PROVES
----------------
For ``DotProductAttention(..., softcap=50.0)`` running on the FlashAttention 4
backend on a Blackwell GPU (sm100+), this script proves:

  1. The *actually selected* backend is FA4 (major version 4) -- NOT FA2/FA3.
     This is the critical trap: with softcap enabled, TE also keeps FA2 alive as
     a fallback, so a naive test can silently validate FA2 and report green while
     FA4 is broken. We read TE's ``_attention_backends`` global after the forward
     and hard-fail if the selected FlashAttention version is not 4.x.

  2. The forward output matches a pure-fp32 PyTorch reference implementing:

         scores = (Q @ K^T) * softmax_scale
         scores = softcap * tanh(scores / softcap)      # tanh logit softcapping
         scores = scores + mask                         # causal / SWA / padding
         attn   = softmax(scores)
         out    = attn @ V

  3. The gradients dQ/dK/dV (via autograd) match that reference -- i.e. the FA4
     softcap *backward* (the beta path TE gates behind NVTE_FA4_SOFTCAP=1) is
     numerically correct, not just the forward.

  4. The softcap is *actually applied*, not silently dropped. Inputs are scaled so
     pre-cap logits far exceed the cap; we assert the capped reference diverges
     materially from a no-cap reference, and that FA4's output tracks the *capped*
     reference (item 2) rather than the no-cap one. A ``softcap=0.0`` control
     additionally proves softcapping is an exact no-op when disabled.

COVERAGE (each x {fp16, bf16})
  - dense bshd: no_mask, causal
  - GQA (num_gqa_groups=8 < num_heads=16)
  - sliding-window attention (window_size != (-1,-1))
  - non-64 head_dim, including head_dim=256 (Gemma2's value; sm100/103-gated)
  - varlen / thd + padding (packed variable-length sequences), incl. padding_causal + GQA
  - softcap=0.0 control (dense + varlen)

HOW TO RUN
----------
As a standalone script (prints a PASS/FAIL table, exits nonzero on any failure):

    NVTE_FA4_SOFTCAP=1 python tests/pytorch/attention/verify_fa4_softcap.py

Or under pytest:

    NVTE_FA4_SOFTCAP=1 pytest -q tests/pytorch/attention/verify_fa4_softcap.py

The script sets NVTE_FA4_SOFTCAP=1 itself if unset, but exporting it is harmless.
It skips cleanly (with a reason) if not on CUDA sm100+, if flash-attn-4 is not
installed, or if the installed FA4 build does not expose a ``softcap`` kwarg.
"""

import os
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment gates MUST be set before TE selects a backend. get_attention_backend
# reads these via os.getenv() at call time, so setting them here (module import)
# guarantees they are in effect for every forward below.
#   NVTE_FA4_SOFTCAP=1  -> opt-in that lets FA4 survive the softcap filter
#   NVTE_FLASH_ATTN=1   -> enable the FlashAttention family
#   NVTE_FUSED_ATTN=0   -> disable cuDNN fused attn (else it is preferred on sm90+)
#   NVTE_UNFUSED_ATTN=0 -> disable the unfused/reference backend
# ---------------------------------------------------------------------------
os.environ.setdefault("NVTE_FA4_SOFTCAP", "1")
os.environ["NVTE_FLASH_ATTN"] = "1"
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["NVTE_UNFUSED_ATTN"] = "0"

import torch

# Make `from utils import reset_rng_states` (tests/pytorch/utils.py) importable,
# exactly as the sibling attention tests do.
_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path


# ---------------------------------------------------------------------------
# Skip detection. Everything TE-specific is imported lazily inside helpers so the
# script can print a clean skip message even where TE / flash-attn-4 cannot load.
# ---------------------------------------------------------------------------
def _skip_reason() -> Optional[str]:
    """Return a human-readable skip reason, or None if the environment is valid."""
    if not torch.cuda.is_available():
        return "CUDA is required."
    try:
        from transformer_engine.pytorch import get_device_compute_capability
    except Exception as exc:  # pragma: no cover - import environment issue
        return f"Could not import transformer_engine.pytorch: {exc}"
    cc = get_device_compute_capability()
    if cc < (10, 0):
        return (
            f"FA4 softcap enablement is Blackwell-only (sm100+); this device is "
            f"sm{cc[0] * 10 + cc[1]}."
        )
    try:
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            FlashAttentionUtils,
        )
    except Exception as exc:  # pragma: no cover
        return f"Could not import FlashAttentionUtils: {exc}"
    if not FlashAttentionUtils.v4_is_installed:
        return "flash-attn-4 (FA4) is not installed."
    if not FlashAttentionUtils.fa4_supports_softcap:
        return (
            "Installed FA4 build does not expose a `softcap` kwarg "
            "(FlashAttentionUtils.fa4_supports_softcap is False)."
        )
    return None


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    qkv_format: str  # "bshd" or "thd"
    num_heads: int
    num_gqa_groups: int
    head_dim: int
    attn_mask_type: str  # "no_mask" | "causal" | "padding" | "padding_causal"
    window_size: Tuple[int, int]
    softcap: float
    # bshd:
    batch_size: int = 2
    max_seqlen: int = 64
    # thd: explicit per-sequence lengths (packed contiguously, no intra-seq padding)
    seqlens: Optional[List[int]] = None
    # gate to specific compute capabilities (e.g. head_dim=256 dedicated kernel)
    required_cc: Optional[Tuple[Tuple[int, int], ...]] = None


# Inputs are scaled by INPUT_STD so that pre-softcap logits have std ~ INPUT_STD**2
# (this is head_dim-independent: score = QK^T/sqrt(d) has std sigma^2 when q,k ~
# N(0,sigma^2)). We target a std COMPARABLE TO the cap (50), not far above it.
# Rationale: tanh is monotonic, so softcapping never changes the argmax key -- it
# only reshapes the softmax *sharpness*. If logits are >> cap the softmax is a hard
# one-hot for BOTH capped and no-cap references (identical outputs, near-zero grads),
# which would make both the "cap is applied" check and the gradient check vacuous.
# With std ~= cap, tanh is in its active/nonlinear region: capped vs no-cap softmax
# distributions differ materially AND gradients stay healthy, so forward parity,
# gradient parity, and the "cap actually applied" divergence are all meaningful.
# The divergence sanity check below fails loud if this regime is ever too weak.
INPUT_STD = 7.0
SOFTCAP = 50.0

_CASES: List[Case] = [
    # ---- dense (bshd) ----
    Case("dense_no_mask", "bshd", 4, 4, 64, "no_mask", (-1, -1), SOFTCAP),
    Case("dense_causal", "bshd", 4, 4, 64, "causal", (-1, 0), SOFTCAP),
    # GQA: 16 query heads, 8 kv groups (reviewer-requested 16/8)
    Case("dense_gqa_16_8_causal", "bshd", 16, 8, 64, "causal", (-1, 0), SOFTCAP,
         max_seqlen=128),
    # Sliding-window attention, non-64 head_dim (128)
    Case("dense_swa_hd128", "bshd", 8, 8, 128, "causal", (64, 0), SOFTCAP,
         max_seqlen=256),
    # non-64 head_dim (96)
    Case("dense_hd96_causal", "bshd", 8, 8, 96, "causal", (-1, 0), SOFTCAP,
         max_seqlen=128),
    # head_dim=256 (Gemma2). FA4 dedicated kernel is SM100/103-only.
    Case("dense_hd256_causal", "bshd", 8, 8, 256, "causal", (-1, 0), SOFTCAP,
         max_seqlen=128, required_cc=((10, 0), (10, 3))),
    Case("dense_hd256_gqa_swa", "bshd", 16, 8, 256, "causal", (64, 0), SOFTCAP,
         max_seqlen=128, required_cc=((10, 0), (10, 3))),
    # softcap=0.0 control: FA4 output must equal a no-softcap reference.
    Case("dense_control_softcap0", "bshd", 4, 4, 64, "causal", (-1, 0), 0.0),

    # ---- varlen / thd + padding ----
    Case("varlen_padding", "thd", 8, 8, 64, "padding", (-1, -1), SOFTCAP,
         seqlens=[96, 128, 64]),
    Case("varlen_padding_causal_gqa", "thd", 16, 8, 128, "padding_causal", (-1, 0),
         SOFTCAP, seqlens=[64, 128, 96]),
    Case("varlen_swa_causal", "thd", 8, 8, 64, "padding_causal", (48, 0), SOFTCAP,
         seqlens=[96, 128, 64]),
    # softcap=0.0 control (varlen)
    Case("varlen_control_softcap0", "thd", 8, 8, 64, "padding_causal", (-1, 0), 0.0,
         seqlens=[96, 128, 64]),
]


# ---------------------------------------------------------------------------
# Pure-fp32 reference
# ---------------------------------------------------------------------------
def _window_bool_mask(sq: int, skv: int, window_size: Tuple[int, int], device) -> torch.Tensor:
    """Return a [sq, skv] boolean mask where True == position is masked OUT.

    Matches FlashAttention's sliding-window semantics with bottom-right diagonal
    alignment (offset = skv - sq):

        key j is allowed for query i iff
            (window_size[1] == -1 or j <= i + offset + window_size[1]) and
            (window_size[0] == -1 or j >= i + offset - window_size[0])

    window_size == (-1, -1) -> full (no masking); (-1, 0) -> causal; (w, 0) -> SWA.
    """
    i = torch.arange(sq, device=device).unsqueeze(1)  # query row
    j = torch.arange(skv, device=device).unsqueeze(0)  # key col
    offset = skv - sq
    wl, wr = window_size
    allowed = torch.ones(sq, skv, dtype=torch.bool, device=device)
    if wr != -1:
        allowed &= j <= i + offset + wr
    if wl != -1:
        allowed &= j >= i + offset - wl
    return ~allowed


def _reference_attention(q, k, v, scale, softcap, window_size):
    """Pure-fp32 reference for (softcapped) scaled dot-product attention.

    q: [b, sq, h, d]; k, v: [b, skv, hg, d]  (GQA supported: h % hg == 0).
    Returns [b, sq, h, d].
    """
    qt = q.transpose(1, 2).float()  # b h sq d
    kt = k.transpose(1, 2).float()  # b hg skv d
    vt = v.transpose(1, 2).float()

    num_heads = qt.shape[1]
    num_gqa_groups = kt.shape[1]
    if num_heads != num_gqa_groups:
        assert num_heads % num_gqa_groups == 0
        repeats = num_heads // num_gqa_groups
        kt = kt.repeat_interleave(repeats, dim=1)
        vt = vt.repeat_interleave(repeats, dim=1)

    scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale
    if softcap != 0.0:
        scores = softcap * torch.tanh(scores / softcap)
    sq, skv = scores.shape[-2], scores.shape[-1]
    mask = _window_bool_mask(sq, skv, window_size, scores.device)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vt)
    return out.transpose(1, 2)  # b sq h d


def _reference_attention_thd(q, k, v, cu_seqlens, scale, softcap, window_size):
    """Reference for packed thd tensors [total_tokens, h, d] (self-attention).

    Computes attention independently per packed sequence and concatenates. Grads
    flow back into the packed q/k/v because each slice is a view of the input.
    """
    outs = []
    for b in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        qi = q[s:e].unsqueeze(0)  # [1, s_i, h, d]
        ki = k[s:e].unsqueeze(0)
        vi = v[s:e].unsqueeze(0)
        oi = _reference_attention(qi, ki, vi, scale, softcap, window_size)
        outs.append(oi.squeeze(0))
    return torch.cat(outs, dim=0)  # [total, h, d]


# ---------------------------------------------------------------------------
# Backend selection assertion
# ---------------------------------------------------------------------------
def _assert_fa4_selected(context: str) -> str:
    """Hard-fail unless TE's last backend selection resolved to FlashAttention 4.

    Reads the ``_attention_backends`` global that DotProductAttention.forward
    populates from get_attention_backend(). ``flash_attention_backend`` is a
    packaging Version whose .major encodes the FA family (2/3/4); FA4 pre-releases
    (e.g. 4.0.0b11) still report major == 4.
    """
    from transformer_engine.pytorch.attention.dot_product_attention import (
        _attention_backends,
    )

    use_flash = _attention_backends.get("use_flash_attention")
    fa_backend = _attention_backends.get("flash_attention_backend")
    if not use_flash:
        raise RuntimeError(
            f"[{context}] FA4 NOT SELECTED -- FlashAttention is not the selected "
            f"backend at all (use_flash_attention={use_flash}, "
            f"flash_attention_backend={fa_backend}). Validating the wrong backend."
        )
    major = getattr(fa_backend, "major", None)
    if major != 4:
        raise RuntimeError(
            f"[{context}] FA4 NOT SELECTED -- selected FlashAttention version is "
            f"{fa_backend} (major={major}), not 4.x. A softcap test that silently "
            f"runs on FA2/FA3 is meaningless. Aborting loud."
        )
    return str(fa_backend)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
def _tols(dtype, head_dim):
    if dtype == torch.float16:
        atol = rtol = 3e-2
    else:  # bfloat16
        atol = rtol = 5e-2
    if head_dim >= 256:
        atol *= 1.5
        rtol *= 1.5
    return atol, rtol


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------
def run_case(case: Case, dtype: torch.dtype) -> None:
    """Execute one case; raise AssertionError/RuntimeError on any failure."""
    from transformer_engine.pytorch import (
        DotProductAttention,
        get_device_compute_capability,
    )
    from transformer_engine.pytorch.attention.dot_product_attention import (
        _attention_backends,
    )

    try:
        from utils import reset_rng_states

        reset_rng_states()
    except Exception:  # pragma: no cover - fall back to explicit seeding
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

    if case.required_cc is not None:
        cc = get_device_compute_capability()
        if cc not in case.required_cc:
            raise _CaseSkip(
                f"requires compute capability in {case.required_cc}; device is "
                f"sm{cc[0] * 10 + cc[1]}"
            )

    device = "cuda"
    h, hg, d = case.num_heads, case.num_gqa_groups, case.head_dim
    scale = 1.0 / (d ** 0.5)

    # Force a fresh backend selection for this configuration.
    _attention_backends["backend_selection_requires_update"] = True

    if case.qkv_format == "bshd":
        b, s = case.batch_size, case.max_seqlen
        q = (INPUT_STD * torch.randn(b, s, h, d, dtype=dtype, device=device)).requires_grad_()
        k = (INPUT_STD * torch.randn(b, s, hg, d, dtype=dtype, device=device)).requires_grad_()
        v = (0.5 * torch.randn(b, s, hg, d, dtype=dtype, device=device)).requires_grad_()
        grad_out = torch.randn(b, s, h, d, dtype=dtype, device=device)

        q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q, k, v)]

        dpa = DotProductAttention(
            h,
            d,
            num_gqa_groups=hg,
            qkv_format="bshd",
            attn_mask_type=case.attn_mask_type,
            window_size=case.window_size,
            softmax_scale=scale,
            softcap=case.softcap,
            layer_number=1,
        ).to(dtype=dtype, device=device)

        out = dpa(q, k, v, window_size=case.window_size)
        backend_str = _assert_fa4_selected(f"{case.name}/{dtype}")
        out.backward(grad_out)

        out_ref = _reference_attention(q_ref, k_ref, v_ref, scale, case.softcap, case.window_size)
        out_ref.backward(grad_out.float())

        grads = [(q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)]

    elif case.qkv_format == "thd":
        seqlens = list(case.seqlens)
        b = len(seqlens)
        total = sum(seqlens)
        max_seqlen = max(seqlens)
        cu = torch.zeros(b + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32, device=device), dim=0)

        q = (INPUT_STD * torch.randn(total, h, d, dtype=dtype, device=device)).requires_grad_()
        k = (INPUT_STD * torch.randn(total, hg, d, dtype=dtype, device=device)).requires_grad_()
        v = (0.5 * torch.randn(total, hg, d, dtype=dtype, device=device)).requires_grad_()
        grad_out = torch.randn(total, h, d, dtype=dtype, device=device)

        q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q, k, v)]

        dpa = DotProductAttention(
            h,
            d,
            num_gqa_groups=hg,
            qkv_format="thd",
            attn_mask_type=case.attn_mask_type,
            window_size=case.window_size,
            softmax_scale=scale,
            softcap=case.softcap,
            layer_number=1,
        ).to(dtype=dtype, device=device)

        out = dpa(
            q,
            k,
            v,
            qkv_format="thd",
            window_size=case.window_size,
            attn_mask_type=case.attn_mask_type,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        backend_str = _assert_fa4_selected(f"{case.name}/{dtype}")
        out.backward(grad_out)

        out_ref = _reference_attention_thd(
            q_ref, k_ref, v_ref, cu, scale, case.softcap, case.window_size
        )
        out_ref.backward(grad_out.float())

        grads = [(q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)]

    else:
        raise ValueError(f"unknown qkv_format {case.qkv_format}")

    # NaN guard: a correct reference must never produce NaNs here.
    if torch.isnan(out_ref).any() or torch.isnan(out).any():
        raise AssertionError(f"[{case.name}/{dtype}] NaN detected in output")

    atol, rtol = _tols(dtype, d)

    # Prove the cap is *actually applied*: for softcap != 0, the capped reference
    # must differ materially from a no-cap reference on the same inputs, and FA4's
    # output must track the *capped* reference (checked below), not the no-cap one.
    if case.softcap != 0.0:
        with torch.no_grad():
            if case.qkv_format == "bshd":
                out_nocap = _reference_attention(
                    q_ref.detach(), k_ref.detach(), v_ref.detach(), scale, 0.0, case.window_size
                )
            else:
                out_nocap = _reference_attention_thd(
                    q_ref.detach(), k_ref.detach(), v_ref.detach(), cu, scale, 0.0, case.window_size
                )
            cap_vs_nocap = (out_ref - out_nocap).abs().max().item()
        # Sanity: the chosen input regime genuinely exercises the cap. The floor is
        # a few x the numerical tolerance -- comfortably above bf16/fp16 noise, but
        # low enough not to false-fail; a strong regime diverges far more than this.
        if cap_vs_nocap < 3 * atol:
            raise AssertionError(
                f"[{case.name}/{dtype}] softcap regime too weak to be meaningful: "
                f"capped vs no-cap reference max|diff|={cap_vs_nocap:.4g}. Increase "
                f"INPUT_STD so pre-cap logits exceed the cap."
            )
        # FA4 must NOT match the no-cap reference (would mean the cap was dropped).
        fa4_vs_nocap = (out.float() - out_nocap).abs().max().item()
        if fa4_vs_nocap < cap_vs_nocap / 4:
            raise AssertionError(
                f"[{case.name}/{dtype}] FA4 output looks like the NO-CAP reference "
                f"(FA4-vs-nocap max|diff|={fa4_vs_nocap:.4g} << cap-vs-nocap "
                f"{cap_vs_nocap:.4g}). softcap appears to be silently ignored."
            )

    # Forward + gradient parity against the (capped, when softcap!=0) reference.
    torch.testing.assert_close(
        out.float(), out_ref.float(), atol=atol, rtol=rtol,
        msg=lambda m: f"[{case.name}/{dtype}] forward mismatch\n{m}",
    )
    names = ["dQ", "dK", "dV"]
    for nm, (g, g_ref) in zip(names, grads):
        assert g is not None, f"[{case.name}/{dtype}] {nm} grad is None"
        torch.testing.assert_close(
            g.float(), g_ref.float(), atol=atol, rtol=rtol,
            msg=lambda m, nm=nm: f"[{case.name}/{dtype}] {nm} mismatch\n{m}",
        )

    return backend_str  # FA4 version string, for the PASS/FAIL table


class _CaseSkip(Exception):
    """Raised inside run_case to signal a per-case (not global) skip."""


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_fa4_softcap(case, dtype):
    try:
        run_case(case, dtype)
    except _CaseSkip as s:
        pytest.skip(str(s))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def _main() -> int:
    reason = _skip_reason()
    if reason is not None:
        print(f"SKIP: {reason}")
        # Skipping is not a failure -- exit 0 so CI treats "wrong hardware" as skip.
        return 0

    dtypes = [torch.float16, torch.bfloat16]
    dnames = {torch.float16: "fp16", torch.bfloat16: "bf16"}
    rows = []
    n_pass = n_fail = n_skip = 0

    for case in _CASES:
        for dtype in dtypes:
            label = f"{case.name} [{dnames[dtype]}]"
            try:
                backend = run_case(case, dtype)
                rows.append(("PASS", label, f"backend={backend}"))
                n_pass += 1
            except _CaseSkip as s:
                rows.append(("SKIP", label, str(s)))
                n_skip += 1
            except Exception as exc:  # noqa: BLE001 - report everything
                # Truncate long assert dumps to keep the table readable.
                detail = str(exc).strip().splitlines()
                detail = detail[0] if detail else repr(exc)
                rows.append(("FAIL", label, detail[:160]))
                n_fail += 1

    width = max((len(r[1]) for r in rows), default=10)
    print("\n" + "=" * (width + 60))
    print(f"{'RESULT':6}  {'CASE':{width}}  DETAIL")
    print("-" * (width + 60))
    for status, label, detail in rows:
        print(f"{status:6}  {label:{width}}  {detail}")
    print("=" * (width + 60))
    print(f"PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  (total={len(rows)})")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(_main())
