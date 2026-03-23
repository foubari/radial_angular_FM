"""Time-conditioned MLP for vector field approximation.

Architecture: identical to the one in 18727.../NN.py for fair comparison.
3 hidden layers, hidden_dim=128, Swish activation, time concatenated at input.
NormalizeLogRadius is kept in preprocessing.py and is NOT used by default.
"""
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


class MLP(nn.Module):
    """Time-conditioned MLP that approximates a vector field v_theta(x, t).

    Args:
        input_dim:  dimension of x
        hidden_dim: width of hidden layers (default 128)
        n_layers:   number of hidden layers (default 3)
        premodule:  None (default) — pass "NormalizeLogRadius" only as explicit ablation
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        premodule: str | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim  # vector field: input_dim = output_dim

        assert premodule in (None, "NormalizeLogRadius", "PolarCoordinatesWithLogRadius"), \
            f"Unknown premodule: {premodule}"
        self.premodule_name = premodule

        if premodule == "NormalizeLogRadius":
            from rafm.models.preprocessing import NormalizeLogRadius
            self.pre = NormalizeLogRadius()
            net_input_dim = input_dim + 1  # unit vector + log norm
        elif premodule == "PolarCoordinatesWithLogRadius":
            from rafm.models.preprocessing import PolarCoordinatesWithLogRadius
            self.pre = PolarCoordinatesWithLogRadius()
            net_input_dim = input_dim
        else:
            self.pre = None
            net_input_dim = input_dim

        # Build MLP: [x_preprocessed || t] -> hidden -> ... -> output
        layers: list[nn.Module] = [nn.Linear(net_input_dim + 1, hidden_dim), Swish()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), Swish()]
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            t: (batch,) or scalar tensor
        Returns:
            v: (batch, input_dim)
        """
        sz = x.shape
        x = x.view(-1, self.input_dim)
        t = t.view(-1, 1).float().expand(x.shape[0], 1)

        if self.pre is not None:
            h = self.pre(x)
        else:
            h = x
        h = torch.cat([h, t], dim=1)
        return self.net(h).view(*sz)
