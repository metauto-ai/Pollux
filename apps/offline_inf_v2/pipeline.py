from io import BytesIO
import numpy as np
from pathlib import Path
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from PIL import Image

from apps.main.modules.preprocess import var_center_crop_size_fn
from apps.offline_inf_v2.data import DataArgs


def decode_image_raw(image_raw):
    try:
        image = Image.open(BytesIO(image_raw)).convert('RGB')
        return np.asarray(image, dtype=np.uint8), np.ones(1, dtype=np.uint8)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return np.zeros((1, 1, 3), dtype=np.uint8), np.zeros(1, dtype=np.uint8)


@pipeline_def
def image_loading_pipeline(_tar_path: str, use_index_files: bool, data_args: DataArgs):
    if use_index_files:
        index_paths = [str(Path(_tar_path).with_suffix(".idx"))]
    else:
        index_paths = []

    images_raw, text, json = fn.readers.webdataset(
        paths=_tar_path,
        index_paths=index_paths,
        ext=["jpg", "txt", "json"],
        missing_component_behavior="error",
    )
    
    # images_gen = fn.experimental.decoders.image(images_raw, device="cpu", output_type=types.RGB)
    images_gen, valid = fn.python_function(images_raw, function=decode_image_raw, device="cpu", num_outputs=2)

    images_gen = fn.resize(
        images_gen.gpu(), 
        device="gpu",
        resize_x=data_args.image_sizes[1], 
        resize_y=data_args.image_sizes[1],
        mode="not_smaller", 
        interp_type=types.DALIInterpType.INTERP_CUBIC
    )

    # get the dynamic crop size
    crop_size = fn.python_function(
        images_gen.shape(device="cpu"),
        data_args.image_sizes[1],
        data_args.patch_size,
        data_args.dynamic_crop_ratio,
        function=var_center_crop_size_fn,
        device="cpu"
    )

    images_gen = fn.crop_mirror_normalize(
        images_gen, 
        device="gpu",
        crop_h=crop_size[0], 
        crop_w=crop_size[1],
        crop_pos_x=0.5, 
        crop_pos_y=0.5, 
        mirror=fn.random.coin_flip(probability=0.5),
        dtype=types.DALIDataType.FLOAT, 
        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255], 
        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        scale=1.0,
    )
    
    images_plan = fn.resize(
        images_gen, 
        device="gpu",
        resize_x=data_args.image_sizes[0], 
        resize_y=data_args.image_sizes[0],
        mode="not_smaller", 
        interp_type=types.DALIInterpType.INTERP_CUBIC
    )

    return images_plan, images_gen, text, json, valid