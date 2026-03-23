"""Standard Euclidean (linear) conditional path for Flow Matching."""
import torch
from torch import Tensor


class EuclideanPath:
    """Linear interpolation between source and target.

    x_t = (1 - t) * x_0 + t * x_1
    dot_x_t = x_1 - x_0  (conditional vector field)
    """
    name = "euclidean"

    def sample_path(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x0: (batch, d) — source samples
            x1: (batch, d) — target samples
            t:  (batch,)   — time in [0, 1]
        Returns:
            x_t: (batch, d)
        """
        t = t.view(-1, 1)
        return (1.0 - t) * x0 + t * x1

    def conditional_vector_field(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Returns:
            u_t: (batch, d) — target for CFM loss
        """
        return x1 - x0
