# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def create_optimizer(model, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    decay_params, nodecay_params = [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # use fused AdamW optimizer by default. 
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,fused=True)
    
    # Print Model Information
    if dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            return parts[0]  # Fallback to first part if no 'module.' prefix
        print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}, betas: {betas}')
        # Number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
        
        print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
        print(f'Optimized parameters: {optim_module_names}')
        print(f'Frozen parameters: {frozen_module_names}')
        
    return optimizer, optimized_param_dict, all_param_dict

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights
    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    # resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")

    return optimizer, lr_scheduler, forward_pass_step, param_update_step


# ===================== LoRA Training Utilities =====================

def create_lora_optimizer(model, weight_decay, learning_rate, betas):
    """
    Create optimizer specifically for LoRA training.
    Only optimizes LoRA parameters and optionally specified modules.

    Args:
        model: The model with LoRA layers applied
        weight_decay: Weight decay for non-LoRA trainable params
        learning_rate: Learning rate
        betas: Adam beta parameters

    Returns:
        optimizer: The AdamW optimizer
        optimized_param_dict: Dict of trainable parameters
    """
    # Get LoRA parameters and other trainable params
    lora_params = []
    other_trainable_params = []
    all_param_dict = {name: param for name, param in model.named_parameters()}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        else:
            other_trainable_params.append(param)

    # LoRA parameters typically don't need weight decay
    optim_groups = []
    if lora_params:
        optim_groups.append({'params': lora_params, 'weight_decay': 0.0, 'lr': learning_rate})

    # Add other trainable params if any (from modules_to_save)
    if other_trainable_params:
        decay_params = [p for p in other_trainable_params if p.dim() > 1]
        nodecay_params = [p for p in other_trainable_params if p.dim() == 1]

        if decay_params:
            optim_groups.append({
                'params': decay_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            })
        if nodecay_params:
            optim_groups.append({
                'params': nodecay_params,
                'weight_decay': 0.0,
                'lr': learning_rate
            })

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

    # Build optimized param dict
    optimized_param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

    # Print info
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_param_count = sum(p.numel() for p in lora_params)

        print(f'LoRA Optimizer: AdamW')
        print(f'Learning rate: {learning_rate}, Weight decay: {weight_decay}, Betas: {betas}')
        print(f'Total parameters: {format_number(total_params)}')
        print(f'Trainable parameters: {format_number(trainable_params)} ({100*trainable_params/total_params:.2f}%)')
        print(f'LoRA parameters: {format_number(lora_param_count)}')

    return optimizer, optimized_param_dict, all_param_dict


def save_lora_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    train_step,
    param_update_step,
    save_path,
    lora_config,
    save_full_model=False,
):
    """
    Save LoRA checkpoint with options for full or LoRA-only weights.

    Args:
        model: The model (potentially wrapped in DDP)
        optimizer: Optimizer state
        lr_scheduler: LR scheduler state
        train_step: Current training step
        param_update_step: Current parameter update step
        save_path: Path to save checkpoint
        lora_config: LoRA configuration (for reproducibility)
        save_full_model: If True, save full model; else save only LoRA weights
    """
    from model.lora import get_lora_state_dict

    if isinstance(model, DDP):
        model_module = model.module
    else:
        model_module = model

    checkpoint = {
        'fwdbwd_pass_step': train_step,
        'param_update_step': param_update_step,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'lora_config': {
            'rank': lora_config.rank,
            'alpha': lora_config.alpha,
            'dropout': lora_config.dropout,
            'target_modules': lora_config.target_modules,
            'bias_trainable': lora_config.bias_trainable,
            'modules_to_save': lora_config.modules_to_save,
        },
    }

    if save_full_model:
        checkpoint['model'] = model_module.state_dict()
        checkpoint['checkpoint_type'] = 'full'
    else:
        checkpoint['lora_state'] = get_lora_state_dict(model_module)
        checkpoint['checkpoint_type'] = 'lora_only'

    torch.save(checkpoint, save_path)


def load_lora_checkpoint(
    load_path,
    model,
    optimizer=None,
    lr_scheduler=None,
    lora_config=None,
):
    """
    Load LoRA checkpoint, applying LoRA if needed.

    Args:
        load_path: Path to checkpoint
        model: Model to load into
        optimizer: Optional optimizer to restore
        lr_scheduler: Optional LR scheduler to restore
        lora_config: LoRA config to use if checkpoint is LoRA-only

    Returns:
        train_step, param_update_step
    """
    from model.lora import apply_lora_to_model, load_lora_state_dict, LoRAConfig

    checkpoint = torch.load(load_path, map_location='cpu')

    if isinstance(model, DDP):
        model_module = model.module
    else:
        model_module = model

    checkpoint_type = checkpoint.get('checkpoint_type', 'full')

    if checkpoint_type == 'lora_only':
        # Apply LoRA structure if not already applied
        stored_config_dict = checkpoint.get('lora_config', {})
        stored_config = LoRAConfig(**stored_config_dict) if stored_config_dict else None
        config_to_use = lora_config or stored_config

        if config_to_use is None:
            raise ValueError("No LoRA config found in checkpoint or provided")

        # Check if LoRA is already applied
        has_lora = any('lora_A' in n for n, _ in model_module.named_parameters())
        if not has_lora:
            apply_lora_to_model(model_module, config_to_use)

        # Load LoRA weights
        load_lora_state_dict(model_module, checkpoint['lora_state'])
    else:
        # Full checkpoint
        model_module.load_state_dict(checkpoint['model'], strict=False)

    # Restore optimizer and scheduler
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            print_rank0("Warning: Could not load optimizer state")

    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except Exception:
            print_rank0("Warning: Could not load lr_scheduler state")

    return checkpoint.get('fwdbwd_pass_step', 0), checkpoint.get('param_update_step', 0)


def auto_resume_lora_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    lora_config,
    reset_training_state=False
):
    """
    Resume LoRA training from the latest checkpoint in the specified directory.

    Args:
        load_path: Directory or checkpoint path
        model: Model with LoRA already applied
        optimizer: Optimizer to restore
        lr_scheduler: LR scheduler to restore
        lora_config: LoRA configuration
        reset_training_state: Whether to reset training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step
    """
    from model.lora import load_lora_state_dict

    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)

    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, starting LoRA training from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, starting from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights (LoRA or full)
    if isinstance(model, DDP):
        model_module = model.module
    else:
        model_module = model

    checkpoint_type = checkpoint.get('checkpoint_type', 'full')

    if checkpoint_type == 'lora_only':
        load_lora_state_dict(model_module, checkpoint['lora_state'])
        print_rank0(f"Loaded LoRA weights from {os.path.abspath(ckpt_path)}")
    else:
        status = model_module.load_state_dict(checkpoint['model'], strict=False)
        print_rank0(f"Loaded full model from {os.path.abspath(ckpt_path)}, status: {status}")

    # Resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except Exception:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")

    return optimizer, lr_scheduler, forward_pass_step, param_update_step


