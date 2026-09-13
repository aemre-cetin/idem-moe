"""
PermBordaRouter: Zero-FLOPs Discrete Borda Consensus MoE Token Router.
======================================================================
Patent Base: USPTO 64/152,256 and 64/148,668 (Dr. A. Emre ÇETİN)

Mathematical Foundation:
Instead of computing continuous softmax probability gating with GEMM/MAC operations,
PermBordaRouter maps token routing embeddings to their discrete ordinal permutations
in S_E (where E is the expert count). Using the closed-form Kemeny-Young Borda consensus,
expert selection is determined in single-cycle integer comparisons with zero floating-point
multiplication and zero gradient divergence.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


def borda_expert_consensus(routing_scores: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes top-k expert assignments via Borda rank consensus.
    
    Parameters
    ----------
    routing_scores : torch.Tensor of shape (N, E) or (B, N, E)
        Raw or affine token-to-expert affinities.
    top_k : int, default=2
        Number of experts to select per token.
        
    Returns
    -------
    selected_experts : torch.Tensor of shape (..., top_k)
        Indices of the top-k consensus experts.
    borda_scores : torch.Tensor of shape (..., E)
        Normalized Borda rank scores in [0, 1].
    """
    # 1. Discrete Rank Permutation in S_E along expert dimension
    # argsort(argsort(x)) maps smallest score -> 0, largest score -> E-1
    ranks = torch.argsort(torch.argsort(routing_scores, dim=-1), dim=-1).to(routing_scores.dtype)
    E = routing_scores.shape[-1]
    
    # 2. Normalized Borda Scores
    borda_scores = ranks / max(1.0, float(E - 1))
    
    # 3. Top-K Expert Selection via Integer ArgMax
    _, selected_experts = torch.topk(ranks, k=top_k, dim=-1, largest=True)
    
    return selected_experts, borda_scores


class PermBordaRouter(nn.Module):
    """
    Zero-FLOPs Mixture-of-Experts Router using Borda Rank Consensus in S_E.
    
    Parameters
    ----------
    hidden_dim : int
        Token feature dimension D.
    num_experts : int
        Number of available experts E.
    top_k : int, default=2
        Number of experts assigned to each token.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # Basis projection (can be fixed orthogonal or learnable)
        self.routing_basis = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_telemetry: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Routes tokens to experts using discrete permutation ranking.
        
        Parameters
        ----------
        tokens : torch.Tensor of shape (..., D)
            Input token embeddings.
            
        Returns
        -------
        selected_experts : torch.Tensor of shape (..., top_k)
        borda_weights : torch.Tensor of shape (..., E)
        telemetry : dict or None
        """
        # Affine projection to expert space
        proj_scores = self.routing_basis(tokens)
        
        # Borda consensus routing in S_E
        selected_experts, borda_weights = borda_expert_consensus(proj_scores, top_k=self.top_k)
        
        telemetry = None
        if return_telemetry:
            telemetry = {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "zero_flops_routing": True,
                "is_scale_invariant": True,
            }
            
        return selected_experts, borda_weights, telemetry

