# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()



# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)

# Check if LoRA inference is enabled
use_lora = config.training.get("use_lora", False)
lora_checkpoint = config.inference.get("lora_checkpoint", "")

if use_lora or lora_checkpoint:
    from model.lora import apply_lora_to_model, load_lora_state_dict, merge_lora_weights, LoRAConfig
    from utils.training_utils import find_checkpoints

    # Determine checkpoint path
    ckpt_path = lora_checkpoint if lora_checkpoint else config.training.checkpoint_dir
    ckpt_paths = find_checkpoints(ckpt_path)

    if ckpt_paths:
        checkpoint = torch.load(ckpt_paths[-1], map_location="cpu")
        checkpoint_type = checkpoint.get('checkpoint_type', 'full')

        if checkpoint_type == 'lora_only':
            # Load base model first if pretrained checkpoint specified
            pretrained_ckpt = config.training.get("pretrained_checkpoint", "")
            if pretrained_ckpt:
                model.load_ckpt(pretrained_ckpt)

            # Create LoRA config from checkpoint or config
            lora_config_dict = checkpoint.get('lora_config', {})
            if lora_config_dict:
                lora_config = LoRAConfig(**lora_config_dict)
            else:
                lora_config = LoRAConfig(
                    rank=config.lora.get("rank", 8),
                    alpha=config.lora.get("alpha", 16.0),
                    dropout=config.lora.get("dropout", 0.0),
                    target_modules=config.lora.get("target_modules", [
                        "attn.to_qkv", "attn.fc", "mlp.mlp.0", "mlp.mlp.2"
                    ]),
                )

            # Apply LoRA structure
            apply_lora_to_model(model, lora_config, verbose=(ddp_info.local_rank == 0))

            # Load LoRA weights
            load_lora_state_dict(model, checkpoint['lora_state'])
            if ddp_info.is_main_process:
                print(f"Loaded LoRA weights from {ckpt_paths[-1]}")

            # Optionally merge for faster inference
            if config.inference.get("merge_lora", True):
                merge_lora_weights(model)
                if ddp_info.is_main_process:
                    print("Merged LoRA weights into base model for efficient inference")
        else:
            # Full checkpoint with LoRA already merged
            model.load_state_dict(checkpoint['model'], strict=False)
            if ddp_info.is_main_process:
                print(f"Loaded full model from {ckpt_paths[-1]}")
    else:
        if ddp_info.is_main_process:
            print(f"No checkpoint found at {ckpt_path}")
else:
    # Standard model loading
    model.load_ckpt(config.training.checkpoint_dir)

model = DDP(model, device_ids=[ddp_info.local_rank])


if ddp_info.is_main_process:  
    print(f"Running inference; save results to: {config.inference_out_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        result = model(batch)
        if config.inference.get("render_video", False):
            result= model.module.render_video(result, **config.inference.render_video_config)
        export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)