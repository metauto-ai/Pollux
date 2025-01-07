"""
* Installation of COSMOS TVAE
```
cd apps/main/modules/Cosmos-Tokenizer
pip3 install -e .
```
* Test
```
cd apps/main
python test_vae.py
```


class LatentVideoVAEArgs:
    model_name: Literal["Hunyuan", "COSMOS-DV", "COSMOS-CV"] = (
        "Hunyuan"  # Default value is "Hunyuan"
    )
    pretrained_model_name_or_path: str = "tencent/HunyuanVideo"
    revision: Optional[str] = None
    variant: Optional[str] = None
    model_dtype: str = "bf16"
    enable_tiling: bool = True
    enable_slicing: bool = True
    
"""

import torch
from modules.vae import LatentVideoVAE, LatentVideoVAEArgs

# Test Hunyuan VAE
hunyuan_config = LatentVideoVAEArgs(
    model_name="Hunyuan",
    pretrained_model_name_or_path="/jfs/checkpoints/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/vae",
)
hunyuan_vae = LatentVideoVAE(hunyuan_config).cuda()
input_tensor = torch.randn(64, 3, 256, 256).cuda()
print("Testing Hunyuan VAE", hunyuan_config.pretrained_model_name_or_path)
print("Input Shape:", input_tensor.shape)
hunyuan_encoded = hunyuan_vae.encode(input_tensor)
print("Latent Shape:", hunyuan_encoded.shape)
hunyuan_reconstructed = hunyuan_vae.decode(hunyuan_encoded)
print("Reconstructed Shape:", hunyuan_reconstructed.shape)
hunyuan_output = hunyuan_vae.forward(input_tensor)
print("Output Shape (Forward Method):", hunyuan_output.shape)
print("==============================")

# Test COSMOS-DV VAE
cosmos_dv_config = LatentVideoVAEArgs(
    model_name="COSMOS-DV",
    pretrained_model_name_or_path="/jfs/checkpoints/cosmos/Cosmos-Tokenizer-DV8x16x16",
)
cosmos_dv_vae = LatentVideoVAE(cosmos_dv_config).cuda()
print("Testing COSMOS-DV VAE", cosmos_dv_config.pretrained_model_name_or_path)
print("Input Shape:", input_tensor.shape)
cosmos_dv_encoded_indices, cosmos_dv_encoded_codes = cosmos_dv_vae.encode(input_tensor)
print("Indices Shape:", cosmos_dv_encoded_indices.shape)
print("Codes Shape:", cosmos_dv_encoded_codes.shape)
cosmos_dv_reconstructed = cosmos_dv_vae.decode(cosmos_dv_encoded_indices)
print("Reconstructed Shape:", cosmos_dv_reconstructed.shape)
cosmos_dv_output = cosmos_dv_vae.forward(input_tensor)
print("Output Shape (Forward Method):", cosmos_dv_output.shape)
print("==============================")

# Test COSMOS-CV VAE
cosmos_cv_config = LatentVideoVAEArgs(
    model_name="COSMOS-CV",
    pretrained_model_name_or_path="/jfs/checkpoints/cosmos/Cosmos-Tokenizer-CV8x16x16",
)
cosmos_cv_vae = LatentVideoVAE(cosmos_cv_config).cuda()
print("Testing COSMOS-CV VAE", cosmos_cv_config.pretrained_model_name_or_path)
print("Input Shape:", input_tensor.shape)
cosmos_cv_encoded = cosmos_cv_vae.encode(input_tensor)
print("Latent Shape:", cosmos_cv_encoded.shape)
cosmos_cv_reconstructed = cosmos_cv_vae.decode(cosmos_cv_encoded)
print("Reconstructed Shape:", cosmos_cv_reconstructed.shape)
cosmos_cv_output = cosmos_cv_vae.forward(input_tensor)
print("Output Shape (Forward Method):", cosmos_cv_output.shape)
print("==============================")
