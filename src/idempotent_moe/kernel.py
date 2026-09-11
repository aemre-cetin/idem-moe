import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False
    class _DummyKernel:
        def __init__(self, fn):
            self.fn = fn
        def __getitem__(self, grid):
            return self
        def __call__(self, *args, **kwargs):
            return None

    class _DummyTriton:
        @staticmethod
        def jit(fn=None, **kwargs):
            if fn is None:
                return lambda f: _DummyKernel(f)
            return _DummyKernel(fn)
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
    triton = _DummyTriton()
    class _DummyTL:
        constexpr = int
        @staticmethod
        def program_id(dim):
            return 0
        @staticmethod
        def arange(start, end):
            return []
        @staticmethod
        def load(ptr):
            return 0
        @staticmethod
        def store(ptr, val):
            pass
    tl = _DummyTL()
@triton.jit
def _inplace_moe_router_compact_kernel(
    Tokens_ptr,         # [E, N, D]
    TargetMap_ptr,      # [E, N]
    stride_ee, stride_en, stride_ed,
    stride_me, stride_mn,
    N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_expert = tl.program_id(0)
    pid_d_blk = tl.program_id(1)

    token_base_ptr = Tokens_ptr + pid_expert * stride_ee
    map_base_ptr   = TargetMap_ptr + pid_expert * stride_me

    offs_d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)

    # Traversal over N candidate tokens
    for i in range(0, N):
        dest_idx = tl.load(map_base_ptr + i * stride_mn)

        # Leader condition: can only be a cycle leader if dest_idx > i
        if dest_idx > i:
            second_hop = tl.load(map_base_ptr + dest_idx * stride_mn)

            if second_hop == i:
                # -------------------------------------------------------------
                # 2-Cycle (Transposition) Fast-Path: In-register token swap
                # -------------------------------------------------------------
                p_i = token_base_ptr + i * stride_en + offs_d * stride_ed
                p_d = token_base_ptr + dest_idx * stride_en + offs_d * stride_ed

                val_i = tl.load(p_i)
                val_d = tl.load(p_d)
                tl.store(p_i, val_d)
                tl.store(p_d, val_i)
            else:
                # -------------------------------------------------------------
                # Generalized Cycle Leader Verification (Orbit length L >= 3)
                # -------------------------------------------------------------
                curr = dest_idx
                is_leader = True
                keep_searching = True

                while keep_searching:
                    if curr == i:
                        keep_searching = False
                    else:
                        if curr < i:
                            is_leader = False
                            keep_searching = False
                        else:
                            curr = tl.load(map_base_ptr + curr * stride_mn)

                if is_leader:
                    p_i = token_base_ptr + i * stride_en + offs_d * stride_ed
                    temp_token = tl.load(p_i)

                    curr_slot = i
                    cycle_active = True

                    while cycle_active:
                        next_slot = tl.load(map_base_ptr + curr_slot * stride_mn)

                        if next_slot == i:
                            tl.store(token_base_ptr + curr_slot * stride_en + offs_d * stride_ed, temp_token)
                            cycle_active = False
                        else:
                            from_ptr = token_base_ptr + next_slot * stride_en + offs_d * stride_ed
                            to_ptr   = token_base_ptr + curr_slot * stride_en + offs_d * stride_ed
                            val = tl.load(from_ptr)
                            tl.store(to_ptr, val)
                            curr_slot = next_slot


def compact_moe_tokens_inplace(
    tokens: torch.Tensor,
    target_map: torch.Tensor,
    block_d: int = 128,
    prefer_native: bool = True
) -> None:
    """
    In-Place Zero-Copy MoE Token Router and Capacity Compactor.
    Dispatches to native C++20 / Blackwell CUDA engine (idempotent-core) or Triton JIT.

    Args:
        tokens:        Tensor of shape [E, N, D] (float16 or float32)
        target_map:    Tensor of shape [E, N] (int32) containing the idempotent permutation map
        block_d:       Tile size along hidden dimension for Triton (default: 128)
        prefer_native: Whether to dispatch to native C++20/CUDA kernel
    """
    assert tokens.is_contiguous(), "Tokens tensor must be contiguous"
    assert target_map.is_contiguous(), "TargetMap tensor must be contiguous"

    if prefer_native:
        try:
            import idempotent_core
            idempotent_core.compact_inplace(tokens, target_map)
            return
        except Exception:
            pass

    E, N, D = tokens.shape
    assert D % block_d == 0, f"Hidden dimension D ({D}) must be divisible by block_d ({block_d})"

    if HAS_TRITON and tokens.is_cuda and _inplace_moe_router_compact_kernel is not None:
        num_d_blocks = D // block_d
        grid = (E, num_d_blocks)
        _inplace_moe_router_compact_kernel[grid](
            tokens, target_map,
            tokens.stride(0), tokens.stride(1), tokens.stride(2),
            target_map.stride(0), target_map.stride(1),
            N=N,
            BLOCK_D=block_d
        )
        return

    # Pure PyTorch in-situ fallback (CPU, macOS MPS, or non-Triton platforms)
    t_map = target_map.view(E, N)
    for e in range(E):
        for i in range(N):
            dest = int(t_map[e, i].item())
            if dest > i and int(t_map[e, dest].item()) == i:
                tmp = tokens[e, i].clone()
                tokens[e, i] = tokens[e, dest]
                tokens[e, dest] = tmp
