from __future__ import annotations

import torch


class EdgeContentPolicy(torch.nn.Module):
    """Policy for choosing answer vs reasoning on a kept AgentPrune edge."""

    ANSWER_ACTION = 0
    REASONING_ACTION = 1

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        initial_answer_bias: float = 1.0,
        initial_step_bias: float = 0.0,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([initial_answer_bias, initial_step_bias]))

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.net(edge_features)
