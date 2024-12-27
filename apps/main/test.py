"""
python -m apps.main.test
"""

import logging
from torchvision.utils import save_image
from apps.main.data import DataLoaderFactory, DataArgs
from apps.main.modules.vae import LatentVideoVAEArgs
from apps.main.modules.schedulers import SchedulerArgs
from apps.main.modules.gen_transformer import PlanTransformerArgs, GenTransformerArgs
from apps.main.modules.tokenizer import TokenizerArgs
from apps.main.model import ModelArgs, Pollux

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for log messages
    handlers=[logging.StreamHandler()],  # Also log to console
)

if __name__ == "__main__":
    # Define model arguments
    vae_arg = LatentVideoVAEArgs(
        pretrained_model_name_or_path="/jfs/checkpoints/Flux-dev"
    )
    scheduler_arg = SchedulerArgs(
        num_train_timesteps=1000,
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        mode_scale=1.29,
        use_dynamic_shifting=True,
    )
    plan_transformer_arg = PlanTransformerArgs(
        dim=2048,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        n_heads=32,
        n_kv_heads=8,
        n_layers=16,
        patch_size=2,
        in_channels=16,
        gen_seqlen=1000,  # for video/image
        attn_type="bi_causal",
        text_seqlen=256,
        vocab_size=128256,
        pre_trained_path="/jfs/checkpoints/Llama-3.2-1B/original/consolidated.00.pth",
    )
    gen_transformer_arg = GenTransformerArgs(
        dim=2048,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        n_heads=32,
        n_kv_heads=8,
        n_layers=16,
        ada_dim=2048,
        patch_size=2,
        in_channels=32,
        out_channels=16,
        tmb_size=256,
        gen_seqlen=1000,
        condition_seqlen=1000,
        pre_trained_path="/jfs/checkpoints/Llama-3.2-1B/original/consolidated.00.pth",
        attn_type="bi_causal",
    )
    tokenizer_arg = TokenizerArgs(
        model_path="/jfs/checkpoints/Llama-3.2-1B/original/tokenizer.model"
    )
    model_arg = ModelArgs(
        gen_transformer=gen_transformer_arg,
        plan_transformer=plan_transformer_arg,
        vae=vae_arg,
        scheduler=scheduler_arg,
        tokenizer=tokenizer_arg,
        text_cfg_ratio=0.1,
        image_cfg_ratio=0.5,
        mask_patch=16,
    )

    # Define data arguments and DataLoaderFactory
    data_config = {
        "data": {
            "preliminary": {
                "ILSVRC/imagenet-1k": {
                    "use": True,
                    "data_name": "ILSVRC/imagenet-1k",
                    "batch_size": 12,
                    "image_size": 256,
                    "num_workers": 8,
                    "split": "train",
                    "root_dir": "/jfs/data",
                    "cache_dir": "/jfs/data/imagenet-1k",
                    "source": "huggingface",
                    "task": "class_to_image",
                }
            }
        }
    }
    dp_rank = 0
    dp_degree = 1
    data_loader_factory = DataLoaderFactory(dp_rank, dp_degree, data_config)
    data_loader = data_loader_factory.create_dataloader("preliminary")

    # Initialize model
    model = Pollux(model_arg)
    model.cuda()

    # Run test
    for batch in data_loader:
        print(batch)
        batch["image"] = batch["image"].cuda()

        # Forward pass
        model(batch)
        break
