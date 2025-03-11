"""
python -m apps.offline_inf_v2.inference.py config=apps/offline_inf_v2/configs/inference.yaml
"""

from dataclasses import dataclass, field
from omegaconf import OmegaConf

from nemo_curator import get_client
from nemo_curator.datasets import ImageTextPairDataset

from apps.offline_inf_v2.data import DataArgs
from apps.offline_inf_v2.model import ModelArgs
from apps.offline_inf_v2.vae_latent_extractor import VAELatentExtractor


@dataclass
class InferenceArgs:
    name: str = field(default="inference")
    model: ModelArgs = field(default_factory=ModelArgs)
    data: DataArgs = field(default_factory=DataArgs)


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(InferenceArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    client = get_client(cluster_type="gpu")

    dataset = ImageTextPairDataset.from_webdataset(
        path=cfg.data.data_path, id_col=cfg.data.id_col
    )

    latent_extractor = VAELatentExtractor(
        model_args=cfg.model,
        data_args=cfg.data,
    )

    dataset_with_latents = latent_extractor(dataset)

    # Metadata will have a new column named "image_latent"
    dataset_with_latents.save_metadata(
        cfg.data.output_path, columns=[cfg.data.id_col, "doc_id", cfg.data.image_latent_column]
    )


if __name__ == "__main__":
    main()
