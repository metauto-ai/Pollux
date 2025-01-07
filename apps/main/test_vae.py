"""
* Installation of COSMOS TVAE
```
cd apps/main/modules/Cosmos-Tokenizer
pip3 install -e .
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



