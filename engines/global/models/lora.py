"""Low-Rank Adaptation (LoRA) for DiffDock-EM finetuning.

LoRA enables efficient finetuning by adding trainable low-rank decomposition
matrices to existing layers while keeping pretrained weights frozen.

Reference: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA adapter for Linear layers.

    Replaces a frozen Linear layer with: output = Wx + (BA)x * (alpha/r)
    where W is the original frozen weight, B and A are trainable low-rank matrices.

    Args:
        original_layer: The Linear layer to wrap
        r: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor (default: 16)
        dropout: Dropout probability on input (default: 0.0)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze the original layer
        original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

        # LoRA matrices: W' = W + BA where B is (out, r) and A is (r, in)
        # A initialized with Kaiming uniform, B initialized to zero
        # This ensures BA = 0 at initialization, preserving pretrained behavior
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Kaiming initialization for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (frozen)
        original_output = self.original_layer(x)

        # LoRA output: (x @ A^T) @ B^T * scaling
        # Equivalent to: x @ (BA)^T * scaling where BA has shape (out, in)
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

        return original_output + lora_output * self.scaling

    def extra_repr(self) -> str:
        return f'r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}'


def get_target_modules_from_config(config: Dict) -> Set[str]:
    """Parse target module names from config.

    Args:
        config: Dict with 'lora_target_modules' key containing module patterns

    Returns:
        Set of module name patterns to target
    """
    target_modules = config.get('lora_target_modules', [
        'conv_layers',
        'rec_emb_layers',
        'final_conv',
        'lig_emb_layers',
    ])

    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(',')]

    return set(target_modules)


def should_apply_lora(name: str, target_modules: Set[str]) -> bool:
    """Check if a parameter should have LoRA applied.

    Args:
        name: Full parameter name (e.g., 'conv_layers.0.linear.weight')
        target_modules: Set of module name patterns to match

    Returns:
        True if LoRA should be applied to this parameter
    """
    for target in target_modules:
        if target in name:
            return True
    return False


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, int, int]:
    """Apply LoRA to target Linear layers in the model.

    This function traverses the model and wraps qualifying Linear layers
    with LoRALinear. Non-target layers and Embedding layers are left unchanged.

    Args:
        model: The model to modify
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        target_modules: Module name patterns to target (default: conv_layers, rec_emb_layers, etc.)
        verbose: Print information about applied LoRA layers

    Returns:
        Tuple of (modified model, number of LoRA layers added, number of LoRA parameters)
    """
    if target_modules is None:
        target_modules = {
            'conv_layers',
            'rec_emb_layers',
            'lig_emb_layers',
            'final_conv',
            'rec_edge_embedding',
            'lig_edge_embedding',
            'cross_edge_embedding',
            'center_edge_embedding',
            'tor_bond_conv',
        }

    lora_layers_added = 0
    lora_params = 0

    # Track modules to replace (can't modify during iteration)
    replacements = []

    def find_linear_layers(module: nn.Module, prefix: str = ''):
        """Recursively find Linear layers that should get LoRA."""
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name

            if isinstance(child, nn.Linear):
                # Check if this layer is in a target module
                if should_apply_lora(full_name, target_modules):
                    replacements.append((module, name, child, full_name))
            else:
                # Recurse into children
                find_linear_layers(child, full_name)

    find_linear_layers(model)

    # Apply replacements
    for parent, name, linear, full_name in replacements:
        lora_layer = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, name, lora_layer)
        lora_layers_added += 1
        lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

        if verbose:
            print(f'  Applied LoRA to: {full_name} '
                  f'({linear.in_features} -> {linear.out_features}, r={r})')

    if verbose:
        print(f'\nLoRA summary: {lora_layers_added} layers, {lora_params:,} trainable params')

    return model, lora_layers_added, lora_params


def freeze_non_lora_params(model: nn.Module, verbose: bool = True) -> Tuple[int, int]:
    """Freeze all parameters except LoRA matrices.

    After calling this, only lora_A and lora_B parameters will be trainable.
    The ring_feature_embedding (if present) is also kept trainable.

    Args:
        model: Model to freeze
        verbose: Print parameter statistics

    Returns:
        Tuple of (total params, trainable params)
    """
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        # Keep LoRA params trainable
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        # Keep ring embedding trainable (new params for EM finetuning)
        elif 'ring_feature_embedding' in name or 'atom_ring_feature_embedding' in name \
                or 'ring_vector_embedding' in name or 'atom_ring_vector_embedding' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    if verbose:
        pct = 100.0 * trainable_params / total_params if total_params > 0 else 0
        print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable ({pct:.2f}%)')

    return total_params, trainable_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model state dict.

    Useful for saving only the finetuned weights (much smaller than full model).

    Args:
        model: Model with LoRA layers

    Returns:
        State dict containing only LoRA and ring embedding parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
        elif ('ring_feature_embedding' in name or 'atom_ring_feature_embedding' in name
              or 'ring_vector_embedding' in name or 'atom_ring_vector_embedding' in name):
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(
    model: nn.Module,
    lora_state: Dict[str, torch.Tensor],
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load LoRA parameters into a model.

    Args:
        model: Model with LoRA layers applied
        lora_state: State dict with LoRA parameters
        strict: Whether to require exact match

    Returns:
        Tuple of (missing keys, unexpected keys)
    """
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []

    for key in lora_state:
        if key in model_state:
            model_state[key] = lora_state[key]
        else:
            unexpected_keys.append(key)

    for key in model_state:
        if 'lora_' in key or 'ring_feature_embedding' in key or 'ring_vector_embedding' in key:
            if key not in lora_state:
                missing_keys.append(key)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f'Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}')

    model.load_state_dict(model_state)
    return missing_keys, unexpected_keys


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the original Linear layers.

    After merging, the model can be used without the LoRA wrapper for faster inference.
    This modifies the model in-place.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with LoRA weights merged (LoRA layers replaced with Linear)
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compute merged weight: W' = W + BA * scaling
            original = module.original_layer
            lora_weight = (module.lora_B @ module.lora_A) * module.scaling

            merged_weight = original.weight.data + lora_weight

            # Create new Linear layer with merged weights
            merged_layer = nn.Linear(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
            )
            merged_layer.weight.data = merged_weight
            if original.bias is not None:
                merged_layer.bias.data = original.bias.data.clone()

            # Replace the LoRA layer with merged Linear
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, attr_name, merged_layer)
            else:
                setattr(model, attr_name, merged_layer)

    return model


def print_trainable_params(model: nn.Module):
    """Print summary of trainable vs frozen parameters."""
    trainable = 0
    frozen = 0

    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)
        else:
            frozen += param.numel()

    total = trainable + frozen
    pct = 100.0 * trainable / total if total > 0 else 0

    print(f'\nTrainable parameters:')
    for name in trainable_names[:20]:  # Show first 20
        print(f'  {name}')
    if len(trainable_names) > 20:
        print(f'  ... and {len(trainable_names) - 20} more')

    print(f'\nTotal: {total:,} params, {trainable:,} trainable ({pct:.2f}%)')
