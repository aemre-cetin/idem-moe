import pytest
import torch
from idempotent_moe import (
    PermBordaRouter,
    borda_expert_consensus,
    InplaceMoERouter,
    generate_idempotent_moe_map,
)


def test_borda_expert_consensus_ranking():
    # 4 tokens, 6 experts
    routing_scores = torch.tensor([
        [0.1, 0.9, 0.3, 0.8, 0.2, 0.4],
        [0.8, 0.1, 0.2, 0.3, 0.7, 0.6],
    ])
    selected_experts, borda_scores = borda_expert_consensus(routing_scores, top_k=2)

    assert selected_experts.shape == (2, 2)
    assert borda_scores.shape == (2, 6)
    # For token 0, highest scores are at index 1 (0.9) and index 3 (0.8)
    assert set(selected_experts[0].tolist()) == {1, 3}
    # For token 1, highest scores are at index 0 (0.8) and index 4 (0.7)
    assert set(selected_experts[1].tolist()) == {0, 4}


def test_borda_scale_invariance():
    """
    Borda consensus must remain identical under monotonic scale transformations.
    """
    scores = torch.randn(10, 8)
    scaled_scores = scores * 250.0 + 100.0  # Monotonic affine shift

    e1, b1 = borda_expert_consensus(scores, top_k=3)
    e2, b2 = borda_expert_consensus(scaled_scores, top_k=3)

    assert torch.equal(e1, e2), "Selected experts must be identical under scaling"
    assert torch.allclose(b1, b2, atol=1e-5), "Normalized Borda scores must match"


def test_perm_borda_router_forward():
    B, N, D = 2, 16, 64
    num_experts = 8
    top_k = 2

    router = PermBordaRouter(hidden_dim=D, num_experts=num_experts, top_k=top_k)
    tokens = torch.randn(B, N, D)
    experts, weights, telemetry = router(tokens, return_telemetry=True)

    assert experts.shape == (B, N, top_k)
    assert weights.shape == (B, N, num_experts)
    assert telemetry is not None
    assert telemetry["zero_flops_routing"] is True


def test_inplace_moe_router_engine_switch():
    """
    Verifies that InplaceMoERouter supports engine='classic' and engine='permnet'.
    """
    E = 4
    N = 16
    D = 32
    C = 8

    router = InplaceMoERouter(hidden_dim=D, num_experts=E, block_d=32)
    tokens = torch.randn(E, N, D)
    scores = torch.rand(E, N)

    # 1. Classic engine
    out_classic = router(tokens.clone(), scores, expert_capacity=C, engine="classic")
    assert out_classic.shape == (E, C, D)

    # 2. PermNet engine
    out_perm = router(tokens.clone(), scores, expert_capacity=C, engine="permnet")
    assert out_perm.shape == (E, C, D)

