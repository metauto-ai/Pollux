"""
conda activate curator
python -m apps.offline_inf_v2.inference.py config=apps/offline_inf_v2/configs/inference.yaml
"""

from dataclasses import dataclass, field
from omegaconf import OmegaConf

import dask_cudf
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


def init_image_text_dataset(cfg: InferenceArgs):
    metadata = dask_cudf.read_parquet(cfg.data.data_path, split_row_groups=False, blocksize=None)
    if 'status' in metadata.columns:
        metadata = metadata[metadata.status != "failed_to_download"]
    metadata = metadata.map_partitions(ImageTextPairDataset._sort_partition, id_col=cfg.data.id_col)
    tar_files = ImageTextPairDataset._get_tar_files(cfg.data.data_path)
    return ImageTextPairDataset(
        path=cfg.data.data_path,
        metadata=metadata,
        tar_files=tar_files,
        id_col=cfg.data.id_col,
    )


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(InferenceArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    assert cfg.data.output_path is not None, f"Output path is required, otherwise the parquets in {cfg.data.data_path} will be overwritten"

    client = get_client(
        cluster_type="gpu",
    )

    print("Inititiating dataset ...")
    dataset = init_image_text_dataset(cfg)
    print("Dataset initialized")

    print("Initializing latent extractor ...")
    latent_extractor = VAELatentExtractor(
        model_args=cfg.model,
        data_args=cfg.data,
        use_index_files=True,
    )
    print("Latent extractor initialized")

    dataset_with_latents = latent_extractor(dataset)

    latent_columns = [
        f"{cfg.data.image_latent_column}_{cfg.data.image_sizes[0]}",
        f"{cfg.data.image_latent_shape_column}_{cfg.data.image_sizes[0]}",
        f"{cfg.data.image_latent_column}_{cfg.data.image_sizes[1]}",
        f"{cfg.data.image_latent_shape_column}_{cfg.data.image_sizes[1]}",
    ]
    
    print("Extracting latents ...")
    # Metadata will have a new column named "image_latent"
    dataset_with_latents.save_metadata(
        cfg.data.output_path, columns=[
            cfg.data.id_col, "_id", cfg.data.caption_column, cfg.data.valid_column, *latent_columns
        ]
    )
    print("Latents extracted")

if __name__ == "__main__":
    main()
