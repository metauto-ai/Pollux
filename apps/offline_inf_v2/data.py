from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArgs:
    data_path: str = field(default="/mnt/pollux/nemo/sample/")
    output_path: Optional[str] = field(default=None)
    enable_checkpointing: Optional[str] = field(default=None)
    id_col: str = field(default="key")
    batch_size: int = field(default=1)
    num_threads_per_worker: int = field(default=4)
    image_size: int = field(default=256)
    patch_size: int = field(default=16)
    dynamic_crop_ratio: float = field(default=1.0)
    image_latent_column: str = field(default="image_latent")
    image_latent_shape_column: str = field(default="image_latent_shape")