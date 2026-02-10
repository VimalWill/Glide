import torch

from torch import nn
from torch import functional as F

class LolcatsHedgehogFeatureMap(nn.Module):
    class FeatureMapMLP(nn.Module):
        def __init__(
            self,
            num_heads: int = 32,
            head_dim: int = 128,
            feature_dim: int = 128,
            dtype: torch.dtype = torch.bfloat16,
            device: torch.device = 'cuda:0',
            skip_connection: bool = False,
            bias: bool = False,
            zero_init: bool = False,
            normal_init: bool = False,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.feature_dim = feature_dim
            self.dtype = dtype
            self.device = device
            self.skip_connection = skip_connection
            self.bias_flag = bias  # Rename to avoid conflict
            self.zero_init = zero_init
            self.normal_init = normal_init

            # Create one Linear layer per head
            self.linears = nn.ModuleList([
                nn.Linear(head_dim, feature_dim, bias=False)
                for _ in range(num_heads)
            ])
            # Move linears to target device/dtype
            # for linear in self.linears:
            #     linear.to(device=device, dtype=dtype)

            # Initialize weights
            self.init_weights_()

            # Post-init adjustments
            if zero_init:
                if skip_connection:
                    self.zero_init_with_skip_()
                else:
                    self.zero_init_()
            if normal_init:
                self.normal_init_()

            # Skip connection check
            if skip_connection:
                assert head_dim == feature_dim, (
                    f"head_dim ({head_dim}) != feature_dim ({feature_dim})"
                )

            # Bias term (shared per head)
            if self.bias_flag:
                self.bias = nn.Parameter(torch.zeros(
                    (1, num_heads, 1, 1),  # Broadcastable shape
                    dtype=dtype, device=device
                ))
                nn.init.kaiming_uniform_(self.bias)
            else:
                self.bias = 0.0

        def init_weights_(self):
            """Initialize Linear weights with kaiming_uniform"""
            for linear in self.linears:
                nn.init.kaiming_uniform_(linear.weight)

        def zero_init_with_skip_(self):
            """Zero all Linear weights"""
            for linear in self.linears:
                nn.init.zeros_(linear.weight)

        def zero_init_(self):
            """Identity initialization when head_dim == feature_dim"""
            if self.head_dim != self.feature_dim:
                raise ValueError("Identity init requires head_dim == feature_dim")
            for linear in self.linears:
                nn.init.eye_(linear.weight)

        def normal_init_(self):
            """Normal initialization for Linear weights"""
            for linear in self.linears:
                nn.init.normal_(linear.weight, std=0.02)

        def forward(self, x: torch.Tensor):
            # Stack weights: (num_heads, out_dim, in_dim)
            weights = torch.stack([l.weight for l in self.linears], dim=0)
            # Transpose to match original einsum format: (h, d, f)
            weights_t = weights.transpose(1, 2)
            # Original computation
            _x = torch.einsum('hdf,bhld->bhlf', weights_t, x) + self.bias
            return x + _x if self.skip_connection else _x

    class ReLU(nn.Module):
        def __init__(self, eps=1e-12):
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor):
            return F.relu(x).clamp(min=self.eps)

    class SoftmaxDim(nn.Module):
        def __init__(self, eps=1e-12):
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor):
            return torch.cat([
                torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
            ], dim=-1).clamp(min=self.eps)

    def __init__(self, head_dim: int, num_heads: int, feature_dim: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.eps = 1e-12
        self.mlp = self.FeatureMapMLP(head_dim=head_dim, num_heads=num_heads, feature_dim=feature_dim)
        self.activation = self.SoftmaxDim(eps=self.eps)

    def forward(self, x):
        return self.activation(self.mlp(x))

class GlideFeatureMap(nn.Module):

    class SoftmaxDim(nn.Module):
        def __init__(self, eps=1e-12):
            super().__init__()
            self.eps = eps
        
        def forward(self, x: torch.Tensor):
            return torch.cat([
                torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
            ], dim=-1).clamp(min=self.eps)
        
    def __init__(self):
        super().__init__()
        self.eps = 1e-12
        self.softmax = self.SoftmaxDim(self.eps)
    
    def forward(self, x):
        return self.softmax(x)
