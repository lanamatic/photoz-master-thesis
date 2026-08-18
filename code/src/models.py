"""
Neural network models for photometric redshift estimation.
 
The baseline NN reproduces Jones et al. (2024a): four hidden layers of 200
units, ReLU, and a skip connection from the input to the final layer.
 
Design notes
------------
- The "skip connection" in Jones concatenates the raw input with the last
  hidden representation before the final linear layer. We implement exactly
  that.
- Only the head differs between models: NN emits one value, MDN emits mixture
  parameters, BNN emits mean and log-variance. The trunk is identical, which
  keeps the comparison fair.
"""
 
from __future__ import annotations
 
import torch
import torch.nn as nn
 
# =============================================================================
#  Shared trunk
# =============================================================================
 
 
class Trunk(nn.Module):
    """
    Four hidden layers of `width`, ReLU, with an input->output skip.
 
    forward(x) returns the concatenation [h, x], where h is the last hidden
    representation. Heads consume this and project to whatever they need. The
    concatenation is the skip connection from Jones et al. (2024a).
    """
 
    def __init__(self, in_dim: int, width: int = 200, depth: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        self.net = nn.Sequential(*layers)
        self.out_dim = width + in_dim   # after concatenating the skip
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return torch.cat([h, x], dim=-1)
 
 
# =============================================================================
#  NN baseline
# =============================================================================
 
 
class NN(nn.Module):
    """
    Point-estimate baseline. Single output, trained with L1 (MAE) loss.
 
    Matches Jones et al. (2024a): MAE loss, Adam, lr 5e-4. 
    Set `dropout > 0`to obtain the MC-Dropout variant.
    """
 
    def __init__(self, in_dim: int = 9, width: int = 200, depth: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.trunk = Trunk(in_dim, width, depth, dropout)
        self.head = nn.Linear(self.trunk.out_dim, 1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)
 
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Point prediction in eval mode (dropout off)."""
        self.eval()
        return self(x)
 
    @torch.no_grad()
    def predict_mc(self, x: torch.Tensor, n_samples: int = 100
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC-Dropout prediction: dropout kept ON at inference.
        
        Runs `n_samples` stochastic forward passes and returns
        (mean, std, samples) where samples has shape (n_samples, N). The std is
        the epistemic uncertainty estimate (Gal & Ghahramani 2016). Requires
        the model to have been built with dropout > 0.
        """
        self.eval()
        dropout_layers = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        if not dropout_layers:
            raise RuntimeError(
                "predict_mc() called on a model with no Dropout layers "
                "(built with dropout=0.0) - std would silently be all zero."
            )
        for m in dropout_layers:
            m.train()
        preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        return preds.mean(0), preds.std(0), preds
 
 
# =============================================================================
#  Loss
# =============================================================================
 
 
def nn_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error, as in Jones et al. (2024a) for the NN."""
    return torch.nn.functional.l1_loss(pred, target)
 
 
# Buidling models
 
def build_model(name: str, in_dim: int = 9, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "nn":
        return NN(in_dim=in_dim, dropout=kwargs.get("dropout", 0.0))
    if name == "mc_dropout":
        return NN(in_dim=in_dim, dropout=kwargs.get("dropout", 0.1))
    # mdn, bnn will be added
    raise ValueError(f"Unknown or not-yet-implemented model: {name!r}")
 
 