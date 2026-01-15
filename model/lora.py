# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# LoRA (Low-Rank Adaptation) implementation for efficient finetuning.

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "attn.to_qkv",
        "attn.fc",
        "mlp.mlp.0",
        "mlp.mlp.2",
    ])
    bias_trainable: bool = False
    modules_to_save: List[str] = field(default_factory=list)


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer that wraps a frozen linear layer.

    Implements: output = frozen_linear(x) + (x @ A @ B) * (alpha / rank)

    Args:
        base_layer: The original nn.Linear layer to wrap
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor for LoRA output
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA decomposition matrices
        # A: down-projection (in_features -> rank)
        # B: up-projection (rank -> out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Optional dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize LoRA weights
        self.reset_lora_parameters()

        # Freeze base layer
        self._freeze_base_layer()

    def reset_lora_parameters(self):
        """Initialize LoRA matrices following the original paper."""
        # Initialize A with Kaiming uniform (He initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros so initial LoRA contribution is zero
        nn.init.zeros_(self.lora_B)

    def _freeze_base_layer(self):
        """Freeze the base layer weights."""
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base layer output (frozen)
        base_output = self.base_layer(x)

        # LoRA path: x @ A @ B * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base layer for inference efficiency.

        Returns:
            New nn.Linear with merged weights
        """
        with torch.no_grad():
            # Compute merged weight: W + (A @ B)^T * scaling
            merged_weight = (
                self.base_layer.weight.data +
                (self.lora_A @ self.lora_B).T * self.scaling
            )

            # Create new linear layer
            merged_layer = nn.Linear(
                self.base_layer.in_features,
                self.base_layer.out_features,
                bias=self.base_layer.bias is not None,
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype,
            )
            merged_layer.weight.data = merged_weight

            if self.base_layer.bias is not None:
                merged_layer.bias.data = self.base_layer.bias.data.clone()

            return merged_layer

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only the LoRA parameters for saving."""
        return {
            'lora_A': self.lora_A.data,
            'lora_B': self.lora_B.data,
        }

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA parameters."""
        self.lora_A.data = state_dict['lora_A']
        self.lora_B.data = state_dict['lora_B']

    def extra_repr(self) -> str:
        return (
            f'in_features={self.base_layer.in_features}, '
            f'out_features={self.base_layer.out_features}, '
            f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}'
        )


def _get_submodule_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by its full name."""
    parts = name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _set_submodule_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by its full name."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)


def _should_apply_lora(name: str, target_patterns: List[str]) -> bool:
    """Check if a layer should have LoRA applied based on name patterns."""
    for pattern in target_patterns:
        if pattern in name:
            return True
    return False


def apply_lora_to_model(
    model: nn.Module,
    lora_config: LoRAConfig,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA adapters to a model in-place.

    Args:
        model: The model to adapt
        lora_config: LoRA configuration
        verbose: Whether to print info about applied LoRA layers

    Returns:
        The modified model (same reference, modified in-place)
    """
    applied_layers = []

    # Find and replace matching linear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if _should_apply_lora(name, lora_config.target_modules):
                # Create LoRA wrapper
                lora_layer = LoRALinear(
                    base_layer=module,
                    rank=lora_config.rank,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                )

                # Replace in model
                _set_submodule_by_name(model, name, lora_layer)
                applied_layers.append((name, module.in_features, module.out_features))

    if verbose:
        print(f"Applied LoRA to {len(applied_layers)} layers:")
        for name, in_f, out_f in applied_layers:
            print(f"  - {name}: ({in_f}, {out_f}) -> rank={lora_config.rank}")

        # Calculate parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model


def freeze_base_model(model: nn.Module, modules_to_save: Optional[List[str]] = None):
    """
    Freeze all parameters except LoRA layers and specified modules.

    Args:
        model: The model with LoRA layers applied
        modules_to_save: List of module name patterns to keep trainable
    """
    modules_to_save = modules_to_save or []

    for name, param in model.named_parameters():
        # Keep LoRA parameters trainable
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            continue

        # Check if in modules_to_save
        should_save = False
        for pattern in modules_to_save:
            if pattern in name:
                should_save = True
                break

        param.requires_grad = should_save


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters for optimizer."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA weights from model state dict."""
    lora_state = {}
    for name, param in model.state_dict().items():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
    """Load LoRA weights into model."""
    model_state = model.state_dict()
    model_state.update(lora_state)
    model.load_state_dict(model_state, strict=False)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base layers for efficient inference.
    Modifies model in-place.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights (no LoRA layers)
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            merged = module.merge_weights()
            _set_submodule_by_name(model, name, merged)

    return model


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count LoRA and total parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'lora_A' in n or 'lora_B' in n
    )

    return {
        'total': total_params,
        'trainable': trainable_params,
        'lora': lora_params,
        'frozen': total_params - trainable_params,
    }
